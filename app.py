"""
Slackbot: Daily Game Leaderboards (Wordle & more)

Features
- Auto-detect Wordle share messages and record scores per workspace + channel + date
- Pluggable parser registry (Wordle included; easy to add others like Connections/Spelling Bee)
- /scoreboard [game] [YYYY-MM-DD] to view daily leaderboard (defaults to today)
- /myscore [game] [YYYY-MM-DD] to view your entries
- /games to list supported games & patterns
- Ties resolved by: fewer guesses better; then earlier submission wins
- SQLite persistence (file path configurable via env)

Quick Start
1) Create a Slack app → Enable Socket Mode OR Events API, add bot token scopes:
   app_mentions:read, channels:history, chat:write, commands, groups:history,
   im:history, mpim:history, reactions:write, users:read
2) Install to workspace and note SLACK_BOT_TOKEN & SLACK_SIGNING_SECRET (and APP_LEVEL_TOKEN if using Socket Mode)
3) Export env vars and run:  python app.py
4) Create slash commands in Slack:
   - /scoreboard  → Request URL: https://YOUR_HOST/slack/events  (or socket mode)
   - /myscore     → same
   - /games       → same

Env Vars
- SLACK_BOT_TOKEN (xoxb-...)
- SLACK_SIGNING_SECRET
- APP_LEVEL_TOKEN (xapp-...)  # if using Socket Mode
- DB_PATH (optional, default './scores.db')
- USE_SOCKET_MODE=true|false (default true)

"""
from __future__ import annotations
import os
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone, date
from typing import Optional, Dict, Tuple, List, Protocol, Iterable

from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from slack_sdk.web.client import WebClient

# -------------------------
# Storage
# -------------------------

DB_PATH = os.getenv("DB_PATH", "./scores.db")

def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

SCHEMA = """
CREATE TABLE IF NOT EXISTS scores(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  workspace_id TEXT NOT NULL,
  channel_id   TEXT NOT NULL,
  user_id      TEXT NOT NULL,
  game_key     TEXT NOT NULL,
  game_label   TEXT NOT NULL,
  game_number  TEXT,
  score_value  INTEGER NOT NULL,  -- smaller is better (e.g., Wordle: 1..6; fail treated as 7)
  raw_score    TEXT NOT NULL,      -- e.g., "4/6", "X/6"
  message_ts   TEXT NOT NULL,      -- Slack timestamp
  message_link TEXT,
  submitted_at TEXT NOT NULL,      -- ISO8601 UTC
  play_date    TEXT NOT NULL       -- YYYY-MM-DD (UTC-based to avoid TZ confusion)
);
CREATE UNIQUE INDEX IF NOT EXISTS uniq_daily_entry
  ON scores(workspace_id, channel_id, user_id, game_key, play_date);
CREATE INDEX IF NOT EXISTS idx_lookup
  ON scores(workspace_id, channel_id, game_key, play_date);
"""

# -------------------------
# Game parser framework
# -------------------------

@dataclass
class ParsedScore:
    game_key: str            # canonical key, e.g., "wordle"
    game_label: str          # display name, e.g., "Wordle"
    game_number: Optional[str]
    score_value: int         # numeric for ranking (lower is better)
    raw_score: str           # as shared, e.g., "3/6" or "X/6"

class GameParser(Protocol):
    key: str
    label: str
    def try_parse(self, text: str) -> Optional[ParsedScore]: ...

class WordleParser:
    key = "wordle"
    label = "Wordle"
    # Example shares:
    # "Wordle 1024 4/6\n\n🟨⬛..." (any grid)
    # "Wordle 250 X/6*" (hard mode has a trailing *)
    WORDLE_LINE = re.compile(r"(?im)^wordle[ 	 ]+(?P<num>[0-9,]+)[ 	 ]+(?P<score>[xX]|[1-6])/(?:6|[0-9])(?:[*])?")

    def try_parse(self, text: str) -> Optional[ParsedScore]:
        m = self.WORDLE_LINE.search(text)
        if not m:
            return None
        num = m.group("num").replace(",", "")
        raw_first = m.group("score").upper()
        raw = f"{raw_first}/6"
        # Fail is X which we map to 7 so it ranks after 6
        score_val = 7 if raw_first == "X" else int(raw_first)
        return ParsedScore(self.key, self.label, num, score_val, raw)

class CluesBySamParser:
    key = "cluesbysam"
    label = "Clues by Sam"

    # Example header:
    # "I solved the daily Clues by Sam (Oct 21st 2025) in 05:51"
    RE_HEADER = re.compile(r"(?im)^\s*I\s+solved\b.*?\bClues by Sam\b.*?$", re.M)
    RE_DATE   = re.compile(r"\((?P<when>[^)]+)\)")
    # Captures h:mm:ss or mm:ss after the word "in"
    RE_TIME   = re.compile(r"(?i)\bin\s+(?P<t>(?:\d{1,2}:)?\d{1,2}:\d{2})\b")

    # Emoji tokens for mistake counting
    TOKENS = {
        "GREEN":  [ "🟩", ":large_green_square:" ],
        "YELLOW": [ "🟨", ":large_yellow_square:" ],
        "HINT":   [ "🟡", ":yellow_circle:" ],
        "DHINT":  [ "🟠", ":orange_circle:", ":large_orange_circle:" ],
    }

    @staticmethod
    def _iso_from_text(dtxt: str) -> Optional[str]:
        # Turn "Oct 21st 2025" -> "2025-10-21" if possible
        try:
            dtxt = re.sub(r"(\d+)(st|nd|rd|th)", r"\1", dtxt.strip(), flags=re.I)
            from datetime import datetime
            return datetime.strptime(dtxt, "%b %d %Y").date().isoformat()
        except Exception:
            return None

    @staticmethod
    def _to_seconds(t: str) -> int:
        # Accept "mm:ss" or "h:mm:ss"
        parts = t.split(":")
        if len(parts) == 2:
            m, s = parts
            return int(m) * 60 + int(s)
        if len(parts) == 3:
            h, m, s = parts
            return int(h) * 3600 + int(m) * 60 + int(s)
        raise ValueError("Bad time format")

    @staticmethod
    def _count_occurrences(text: str, needles: list[str]) -> int:
        total = 0
        for n in needles:
            total += text.count(n)
        return total

    def try_parse(self, text: str) -> Optional[ParsedScore]:
        if not text or "clues by sam" not in text.lower():
            return None
        if not self.RE_HEADER.search(text):
            return None

        # 1) Parse time (required)
        mt = self.RE_TIME.search(text)
        if not mt:
            return None
        time_str = mt.group("t")
        seconds = self._to_seconds(time_str)  # lower is better within same mistake count

        # 2) Count mistakes per your guide
        mistakes  = 0
        mistakes += self._count_occurrences(text, self.TOKENS["YELLOW"]) * 1
        mistakes += self._count_occurrences(text, self.TOKENS["HINT"])   * 1
        mistakes += self._count_occurrences(text, self.TOKENS["DHINT"])  * 2
        # Greens counted as 0 (we don't add them)

        # 3) Optional puzzle identifier from date in parentheses
        md = self.RE_DATE.search(text)
        game_number = None
        if md:
            iso = self._iso_from_text(md.group("when"))
            game_number = iso or md.group("when")

        # 4) Composite score: mistakes first, then time
        score_value = mistakes * 1_000_000 + seconds  # lower is better

        # 5) Raw string for display
        raw = f"{mistakes} mistake{'s' if mistakes != 1 else ''} • {time_str}"

        return ParsedScore(self.key, self.label, game_number, score_value, raw)

class TravleParser:
    key = "travle"
    label = "Travle"
    # Supports two common share formats:
    # A) Hashtag + plus-score   → "#travle #1011 +3"
    # B) Header with fraction   → "Travle #1011 3/6"
    LINE_HASHTAG_PLUS = re.compile(
        r"(?im)^\s*#?travle\b[^\n]*?#(?P<num>[\d,]+)\s*\+(?P<score>\d+)\b"
    )
    LINE_FRACTION = re.compile(
        r"(?im)^\s*travle[ \t\u00A0#]+(?P<num>[\d,]+)[ \t\u00A0]+(?P<score>[xX]|\d+)\/(?:\d+)"
    )

    def try_parse(self, text: str) -> Optional[ParsedScore]:
        # Try hashtag + plus score first (e.g., "#travle #1011 +3")
        m = self.LINE_HASHTAG_PLUS.search(text)
        if m:
            num = m.group("num").replace(",", "")
            n = int(m.group("score"))
            raw = f"+{n}"         # store raw as "+3"
            score_val = n         # lower is better
            return ParsedScore(self.key, self.label, num, score_val, raw)

        # Fallback to the fraction style (e.g., "Travle #1011 3/6")
        m = self.LINE_FRACTION.search(text)
        if m:
            num = m.group("num").replace(",", "")
            raw_first = m.group("score").upper()
            raw = f"{raw_first}/6" if raw_first != "X" else "X/6"
            score_val = 99 if raw_first == "X" else int(raw_first)
            return ParsedScore(self.key, self.label, num, score_val, raw)

        return None

class FoodguessrParser:
    key = "foodguessr"
    label = "FoodGuessr"

    # Matches the common header line:
    #   "I got 11,500 on the FoodGuessr Daily!"
    RE_HEADER = re.compile(
        r"(?im)^\s*I\s+got\s+(?P<pts>[\d,]+)\s+on\s+the\s+FoodGuessr\b"
    )

    # Fallback: find round lines and sum them:
    #   "🌕🌕🌕🌑 3,500 (Round 1)"
    RE_ROUND_LINE = re.compile(
        r"(?im)^\s*.*?(?P<rpts>[\d,]+)\s*\(\s*Round\s*\d+\s*\)\s*$"
    )

    # Optional: generic points mention like "FoodGuessr ... 11500 pts"
    RE_GENERIC = re.compile(
        r"(?im)\bFoodGuessr\b[^\n]*?(?P<pts>[\d,]+)\s*(?:pts|points)\b"
    )

    def try_parse(self, text: str) -> Optional[ParsedScore]:
        if not text or "foodguessr" not in text.lower():
            return None

        # normalize NBSPs/commas are handled when converting to int
        # 1) Header "I got <pts> on the FoodGuessr ..."
        m = self.RE_HEADER.search(text)
        if m:
            pts = int(m.group("pts").replace(",", ""))
            return ParsedScore(self.key, self.label, None, -pts, f"{pts} pts")

        # 2) Sum per-round points if present
        rounds = [int(x.replace(",", "")) for x in self.RE_ROUND_LINE.findall(text)]
        if rounds:
            pts = sum(rounds)
            return ParsedScore(self.key, self.label, None, -pts, f"{pts} pts")

        # 3) Generic “... 11500 pts/points” after the word FoodGuessr
        g = self.RE_GENERIC.search(text)
        if g:
            pts = int(g.group("pts").replace(",", ""))
            return ParsedScore(self.key, self.label, None, -pts, f"{pts} pts")

        return None


class GeoGridParser:
    key = "geogrid"
    label = "GeoGrid"

    RE_SCORE = re.compile(r"(?im)Score:\s*(?P<score>\d+(?:\.\d+)?)")
    RE_RANK  = re.compile(r"(?im)Rank:\s*(?P<cur>[\d,]+)\s*/\s*(?P<tot>[\d,]+)")
    RE_BOARD = re.compile(r"(?im)(?:Board|Game)\s*#\s*(?P<num>[\d,]+)")

    def try_parse(self, text: str) -> Optional[ParsedScore]:
        if not text or ("geogrid" not in text.lower() and "geogridgame" not in text.lower()):
            return None

        m_score = self.RE_SCORE.search(text)
        if not m_score:
            return None

        # Numeric score for ranking (lower = better)
        score_float = float(m_score.group("score"))
        score_value = int(round(score_float * 10))  # keep 1 decimal precision as int

        # Game number (board ID)
        m_board = self.RE_BOARD.search(text)
        game_number = m_board.group("num").replace(",", "") if m_board else None

        # Build raw string exactly as in share (Score + Rank if present)
        m_rank = self.RE_RANK.search(text)
        if m_rank:
            raw = f"Score: {score_float:g} | Rank: {m_rank.group('cur')}/{m_rank.group('tot')}"
        else:
            raw = f"Score: {score_float:g}"

        return ParsedScore(self.key, self.label, game_number, score_value, raw)

class ConnectionsParser:
    key = "connections"
    label = "Connections"

    # Typical share:
    # Connections
    # Puzzle #836
    # 🟦🟦🟦🟦
    # 🟨🟨🟨🟨
    # 🟩🟩🟩🟩
    # 🟪🟪🟪🟪
    # (sometimes includes “Mistakes: 2/4”)

    RE_HEADER   = re.compile(r"(?im)^\\s*connections\\s*$")
    RE_PUZZLE   = re.compile(r"(?im)^puzzle\\s*#\\s*(?P<num>[\\d,]+)")
    RE_MISTAKES = re.compile(r"(?im)mistakes?\\s*:\\s*(?P<m>\\d+)\\s*/\\s*4")

    def try_parse(self, text: str) -> Optional[ParsedScore]:
        if not text:
            return None
        if not self.RE_HEADER.search(text):
            return None

        # Puzzle number (optional)
        m_puz = self.RE_PUZZLE.search(text)
        game_number = m_puz.group("num").replace(",", "") if m_puz else None

        # Mistakes if present; lower is better. Default to 0 if no mistakes line.
        m_mis = self.RE_MISTAKES.search(text)
        mistakes = int(m_mis.group("m")) if m_mis else 0

        # Raw string for display
        raw = f"{mistakes} mistake{'s' if mistakes != 1 else ''}" if m_mis else "Solved"

        # Ranking: fewer mistakes = better
        score_value = mistakes

        return ParsedScore(self.key, self.label, game_number, score_value, raw)

# Registry for future games (NYT Connections, Spelling Bee, etc.)
PARSERS: List[GameParser] = [
    WordleParser(),
    TravleParser(),
    FoodguessrParser(),
    GeoGridParser(),
    ConnectionsParser(),  
    CluesBySamParser(),   # ← add this

]

# Slack app
# -------------------------

use_socket_mode = os.getenv("USE_SOCKET_MODE", "true").lower() == "true"
app = App(token=os.getenv("SLACK_BOT_TOKEN"), signing_secret=os.getenv("SLACK_SIGNING_SECRET"))

# Ensure DB schema
with _get_conn() as c:
    c.executescript(SCHEMA)

# Utilities

def _utc_today_str() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def _play_date_from_ts(ts: str) -> str:
    # Slack ts is like "1726812345.001200" seconds.
    try:
        seconds = float(ts)
        d = datetime.fromtimestamp(seconds, tz=timezone.utc).date()
    except Exception:
        d = date.today()
    return d.isoformat()


def _permalink(client: WebClient, channel: str, ts: str) -> Optional[str]:
    try:
        resp = client.chat_getPermalink(channel=channel, message_ts=ts)
        return resp.get("permalink")
    except Exception:
        return None


def parse_any_score(text: str) -> Optional[ParsedScore]:
    for p in PARSERS:
        res = p.try_parse(text)
        if res:
            return res
    return None


def save_score(client: WebClient, team_id: str, channel_id: str, user_id: str, ts: str, parsed: ParsedScore) -> Tuple[bool, str]:
    play_date = _play_date_from_ts(ts)
    link = _permalink(client, channel_id, ts)
    with _get_conn() as c:
        try:
            c.execute(
                """
                INSERT INTO scores(workspace_id, channel_id, user_id, game_key, game_label, game_number,
                                   score_value, raw_score, message_ts, message_link, submitted_at, play_date)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    team_id, channel_id, user_id, parsed.game_key, parsed.game_label, parsed.game_number,
                    parsed.score_value, parsed.raw_score, ts, link,
                    datetime.now(timezone.utc).isoformat(), play_date,
                ),
            )
            return True, play_date
        except sqlite3.IntegrityError:
            # Update existing (user reposted or improved?) Keep best score and earliest time.
            cursor = cursor = c.execute(
                """
                SELECT score_value, submitted_at FROM scores
                WHERE workspace_id=? AND channel_id=? AND user_id=? AND game_key=? AND play_date=?
                """,
                (team_id, channel_id, user_id, parsed.game_key, play_date),
            )
            row = cursor.fetchone()
            if row:
                existing_score = row[0]
                existing_time = row[1]
                improved = parsed.score_value < existing_score
                if improved:
                    c.execute(
                        """
                        UPDATE scores SET score_value=?, raw_score=?, game_number=?, message_ts=?, message_link=?, submitted_at=?
                        WHERE workspace_id=? AND channel_id=? AND user_id=? AND game_key=? AND play_date=?
                        """,
                        (
                            parsed.score_value, parsed.raw_score, parsed.game_number, ts, link,
                            datetime.now(timezone.utc).isoformat(),
                            team_id, channel_id, user_id, parsed.game_key, play_date,
                        ),
                    )
                    return True, play_date
            return False, play_date


# -------------------------
# Event listeners
# -------------------------

@app.event("message")
def handle_messages(event, client, logger, body, say):
    # Ignore bot messages/edits/threads except normal user posts
    subtype = event.get("subtype")
    if subtype is not None:
        return
    text = event.get("text", "") or ""
    ts = event.get("ts")
    channel = event.get("channel")
    user = event.get("user")
    team_id = body.get("team_id") or body.get("authorizations", [{}])[0].get("team_id")

    parsed = parse_any_score(text)
    if not parsed:
        return

    ok, play_date = save_score(client, team_id, channel, user, ts, parsed)
    if ok:
        try:
            client.reactions_add(channel=channel, timestamp=ts, name="trophy")
        except Exception:
            pass
        say(thread_ts=ts, text=f"Recorded {parsed.game_label} for {play_date}: *{parsed.raw_score}* — thanks!")
    else:
        say(thread_ts=ts, text=f"You already have a {parsed.game_label} entry for {play_date}. If you beat your score, repost and I'll update it.")


# -------------------------
# Slash commands
# -------------------------

@app.command("/today")
def today(ack, respond, command):
    """
    Usage: /today [@user]
    Shows today's leaderboards for all games with entries in this channel.
    If @user is provided, shows only games that user posted a score for today.
    """
    ack()
    text = command.get("text", "") or ""
    m = re.search(r"<@([A-Z0-9]+)>", text)
    target_user = m.group(1) if m else None
    day = _utc_today_str()
    team_id = command["team_id"]
    channel_id = command["channel_id"]

    with _get_conn() as c:
        if target_user:
            games = c.execute(
                """
                SELECT DISTINCT game_key, game_label
                FROM scores
                WHERE workspace_id=? AND channel_id=? AND play_date=? AND user_id=?
                ORDER BY game_label
                """,
                (team_id, channel_id, day, target_user),
            ).fetchall()
        else:
            games = c.execute(
                """
                SELECT DISTINCT game_key, game_label
                FROM scores
                WHERE workspace_id=? AND channel_id=? AND play_date=?
                ORDER BY game_label
                """,
                (team_id, channel_id, day),
            ).fetchall()

    if not games:
        suffix = f" for <@{target_user}>" if target_user else ""
        respond(response_type="in_channel", text=f"📭 No scoreboards for {day}{suffix} yet.")
        return

    blocks = []
    title = f"📊 Leaderboards for *{day}*" + (f" — for <@{target_user}> 🧑‍💻" if target_user else "")
    blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": title}})

    # Emoji per game
    emoji_by_game = {
        "wordle": "🟩",
        "travle": "🗺️",
        "foodguessr": "🍜",
        "geogrid": "🌍",
    }

    with _get_conn() as c:
        for g in games:
            gkey = g["game_key"]
            glabel = g["game_label"]
            rows = c.execute(
                """
                SELECT * FROM scores
                WHERE workspace_id=? AND channel_id=? AND game_key=? AND play_date=?
                ORDER BY score_value ASC, submitted_at ASC
                """,
                (team_id, channel_id, gkey, day),
            ).fetchall()

            header_emoji = emoji_by_game.get(gkey, "🏆")
            blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": f"{header_emoji} *{glabel}*"}})
            # Use medalized renderer
            boards_text = _render_board_with_medals(rows)
            blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": boards_text}})
            blocks.append({"type": "divider"})

    # Broadcast to everyone in the channel
    respond(response_type="in_channel", blocks=blocks)

# -------------------------

def _render_board(rows: Iterable[sqlite3.Row]) -> str:
    lines = ["*Rank*  *User*          *Score*    *When*"]
    rank = 1
    for r in rows:
        user = f"<@{r['user_id']}>"
        score = r["raw_score"]
        when = r["submitted_at"].replace("T", " ").split(".")[0] + "Z"
        link = r["message_link"]
        if link:
            lines.append(f"{rank:>2}. {user:<14} {score:<8}  <{link}|permalink>")
        else:
            lines.append(f"{rank:>2}. {user:<14} {score:<8}  {when}")
        rank += 1
    if rank == 1:
        return "No entries yet. Share a result message to join today’s board!"
    return "\n".join(lines)


def _render_board_with_medals(rows: Iterable[sqlite3.Row]) -> str:
    """Like _render_board, but decorates top 3 with 🥇🥈🥉."""
    lines = ["*Rank*  *User*          *Score*    *When*"]
    medal_for = {1: "🥇", 2: "🥈", 3: "🥉"}
    rank = 1
    for r in rows:
        user = f"<@{r['user_id']}>"
        score = r["raw_score"]
        when = r["submitted_at"].replace("T", " ").split(".")[0] + "Z"
        link = r["message_link"]
        medal = medal_for.get(rank, f"{rank:>2}.")
        prefix = f"{medal}" if medal in ("🥇", "🥈", "🥉") else medal  # keep alignment for non-medal rows
        if link:
            lines.append(f"{prefix} {user:<14} {score:<10}  <{link}|permalink>")
        else:
            lines.append(f"{prefix} {user:<14} {score:<10}  {when}")
        rank += 1
    if rank == 1:
        return "No entries yet. Share a result message to join today’s board!"
    return "\n".join(lines)

def _parse_args(text: str) -> Tuple[str, Optional[str]]:
    parts = [p for p in text.strip().split() if p]
    game = parts[0].lower() if parts else "wordle"
    day = parts[1] if len(parts) > 1 else _utc_today_str()
    return game, day


@app.command("/scoreboard")
def scoreboard(ack, respond, command):
    ack()
    game, day = _parse_args(command.get("text", ""))
    with _get_conn() as c:
        rows = c.execute(
            """
            SELECT * FROM scores
            WHERE workspace_id=? AND channel_id=? AND game_key=? AND play_date=?
            ORDER BY score_value ASC, submitted_at ASC
            """,
            (command["team_id"], command["channel_id"], game, day),
        ).fetchall()
    title = f"{game.title()} leaderboard for {day}"
    respond(blocks=[
        {"type": "section", "text": {"type": "mrkdwn", "text": f"*{title}*"}},
        {"type": "section", "text": {"type": "mrkdwn", "text": _render_board(rows)}},
    ])


@app.command("/myscore")
def myscore(ack, respond, command):
    ack()
    game, day = _parse_args(command.get("text", ""))
    with _get_conn() as c:
        row = c.execute(
            """
            SELECT * FROM scores
            WHERE workspace_id=? AND channel_id=? AND game_key=? AND play_date=? AND user_id=?
            """,
            (command["team_id"], command["channel_id"], game, day, command["user_id"]),
        ).fetchone()
    if not row:
        respond(f"No {game.title()} entry for you on {day} yet. Share your result to record it!")
    else:
        respond(f"Your {game.title()} on {day}: *{row['raw_score']}* (posted at {row['submitted_at']})")


@app.command("/games")
def games(ack, respond):
    ack()
    lines = ["*Supported games:*"]
    for p in PARSERS:
        if isinstance(p, WordleParser):
            pattern = "Wordle <#> <n/6 or X/6>"
        elif isinstance(p, TravleParser):
            pattern = "Travle #<puzzle> <n/?>"
        elif isinstance(p, FoodguessrParser):
            pattern = "“I got <points> on the FoodGuessr Daily!” or per-round lines"
        elif isinstance(p, GeoGridParser):
            pattern = "Includes “Score: <float> | Rank: x/y” and “Board #<id>”"
        elif isinstance(p, ConnectionsParser):
            pattern = "Includes “Connections” header; optional “Puzzle #<id>” and “Mistakes: n/4”"  
        elif isinstance(p, CluesBySamParser):
            pattern = '“I solved the daily Clues by Sam (Mon Dd YYYY) in mm:ss”'
        else:
            pattern = "See docs"
        lines.append(f"• *{p.label}* (`{p.key}`) — share pattern: `{pattern}`")
    respond("\n".join(lines))


@app.command("/leaderboards")
def leaderboards(ack, respond, command):
    """
    Usage: /leaderboards
    Shows all-time power rankings (average daily percentile) for ALL games
    in the current channel. Percentile per day = 1.0 for 1st, 0.0 for last;
    solo day counts as 1.0. Higher is better.
    """
    ack()
    team_id = command["team_id"]
    channel_id = command["channel_id"]

    # Title block
    blocks = [
        {"type": "section", "text": {"type": "mrkdwn", "text": "*All-time leaderboards*"}}
    ]

    # One section per game in PARSERS
    for parser in PARSERS:
        game = parser.key
        glabel = parser.label

        # All-time power ranking for this game in this channel
        sql = """
        WITH base AS (
          SELECT workspace_id, channel_id, user_id, game_key, play_date,
                 score_value, submitted_at
          FROM scores
          WHERE workspace_id=? AND channel_id=? AND game_key=?
        ),
        ranked AS (
          SELECT user_id, play_date,
                 ROW_NUMBER() OVER (
                   PARTITION BY play_date
                   ORDER BY score_value ASC, submitted_at ASC
                 ) AS rn,
                 COUNT(*) OVER (PARTITION BY play_date) AS cnt
          FROM base
        ),
        percentiles AS (
          SELECT user_id, play_date,
                 CASE
                   WHEN cnt <= 1 THEN 1.0
                   ELSE 1.0 - ((rn - 1.0) / (cnt - 1.0))
                 END AS p
          FROM ranked
        ),
        agg AS (
          SELECT user_id,
                 COUNT(*) AS days_played,
                 AVG(p)   AS avg_pct
          FROM percentiles
          GROUP BY user_id
        )
        SELECT user_id, days_played, avg_pct
        FROM agg
        ORDER BY avg_pct DESC, days_played DESC;
        """

        with _get_conn() as c:
            rows = c.execute(sql, (team_id, channel_id, game)).fetchall()

        if not rows:
            # No history for this game in this channel; skip section
            continue

        # Build a small text leaderboard for this game
        lines = [
            f"*{glabel}*",
            "*Rank*  *User*           *Avg pct*  *Days played*"
        ]
        rank = 1
        for r in rows:
            user = f"<@{r['user_id']}>"
            avg_pct = f"{float(r['avg_pct']):.3f}"
            days = int(r["days_played"])
            lines.append(f"{rank:>2}. {user:<15} {avg_pct:>8}   {days:>12}")
            rank += 1

        blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": "\n".join(lines)}})
        blocks.append({"type": "divider"})

    if len(blocks) == 1:
        respond("No history yet in this channel.")
    else:
        # Remove trailing divider for neatness
        if blocks[-1]["type"] == "divider":
            blocks.pop()
        respond(blocks=blocks)

# -------------------------
# Local dev server (Slash Commands via HTTP) or Socket Mode worker
# -------------------------

if __name__ == "__main__":
    if use_socket_mode:
        app_token = os.getenv("APP_LEVEL_TOKEN")
        if not app_token:
            raise SystemExit("APP_LEVEL_TOKEN is required when USE_SOCKET_MODE=true")
        print("→ Starting Slack bot in Socket Mode…")
        SocketModeHandler(app, app_token).start()
    else:
        from slack_bolt.adapter.fastapi import SlackRequestHandler
        import uvicorn
        from fastapi import FastAPI, Request

        api = FastAPI()
        handler = SlackRequestHandler(app)

        @api.post("/slack/events")
        async def slack_events(req: Request):
            return await handler.handle(req)

        print("→ Starting HTTP server on :3000 …")
        uvicorn.run(api, host="0.0.0.0", port=3000)



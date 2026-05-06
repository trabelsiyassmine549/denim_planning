"""
utils/time_utils.py — Productive-minute time model
===================================================
Work hours: 00h00 to 00h00 (midnight-to-midnight) — 24 hours/day, 7 days/week.

ENCODING — "productive minutes" (PM):
  PM 0    = 00h00  day 0
  PM 1439 = 23h59  day 0
  PM 1440 = 00h00  day 1
  ...

Constants:
  PPD = 1440   # 24 hours x 60 minutes per day
"""

from datetime import date, datetime, timedelta

PPD            = 1440   # 24 hours x 60 minutes
DAY_START_HOUR = 0
DAY_HOURS      = 24

# Legacy aliases kept for backward compatibility
MORN_MINS         = PPD
AFTN_MINS         = 0
LUNCH_START       = PPD
WORK_MINS_PER_DAY = PPD

JOURS_FR = ["Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim"]

# ---------------------------------------------------------------------------
# START_DATE — always the current date at the moment it is accessed.
# This ensures that when the user clicks "Run", the schedule is anchored
# to today's date, not to the date when the server process started.
# ---------------------------------------------------------------------------

class _TodayProxy:
    """
    A lazy date proxy: evaluates date.today() on every attribute/operation access.
    Drop-in replacement for a plain `date` object in all arithmetic and
    comparison expressions used across the codebase.
    """
    def __getattr__(self, name):
        return getattr(date.today(), name)

    def __add__(self, other):
        return date.today() + other

    def __radd__(self, other):
        return other + date.today()

    def __sub__(self, other):
        return date.today() - other

    def __rsub__(self, other):
        return other - date.today()

    def __eq__(self, other):
        return date.today() == other

    def __lt__(self, other):
        return date.today() < other

    def __le__(self, other):
        return date.today() <= other

    def __gt__(self, other):
        return date.today() > other

    def __ge__(self, other):
        return date.today() >= other

    def isoformat(self):
        return date.today().isoformat()

    def __str__(self):
        return date.today().isoformat()

    def __repr__(self):
        return f"_TodayProxy({date.today().isoformat()})"

    def weekday(self):
        return date.today().weekday()


START_DATE = _TodayProxy()


# ---------------------------------------------------------------------------
# Calendar helpers
# ---------------------------------------------------------------------------

def working_day_date(offset_days: int) -> date:
    """
    Return the calendar date `offset_days` days from today.
    Evaluated at call time so the schedule is always anchored to the
    current date when the optimisation is triggered.
    No weekend skipping — the workshop runs 24/7.
    """
    return date.today() + timedelta(days=max(offset_days, 0))


def date_to_day_offset(iso_date: str) -> int:
    """
    Convert an ISO date string to a calendar-day offset from today.
    Evaluated at call time — the deadline is relative to the exact
    moment the optimisation is triggered, not server-start time.
    Returns minimum 1 so deadlines today or in the past still get
    a non-zero PM deadline for the solver.
    No weekend skipping — the workshop runs 24/7.
    """
    target = date.fromisoformat(iso_date)
    delta  = (target - date.today()).days
    return max(delta, 1)


# Legacy alias used by Diagnostic.py
def date_to_offset(iso_date: str) -> int:
    return date_to_day_offset(iso_date)


# ---------------------------------------------------------------------------
# Productive-minute <-> clock conversion
# ---------------------------------------------------------------------------

def pm_to_clock(pm: int) -> tuple:
    """
    Convert a productive-minute value to (day_offset, hour, minute).

    Examples:
      pm=0    -> (0,  0,  0)   00h00 day 0
      pm=1439 -> (0, 23, 59)   23h59 day 0
      pm=1440 -> (1,  0,  0)   00h00 day 1
    """
    day = pm // PPD
    off = pm % PPD
    h   = off // 60
    m   = off % 60
    return day, h, m


def pm_to_hhmm(pm: int) -> str:
    _, h, m = pm_to_clock(pm)
    return f"{h:02d}h{m:02d}"


def pm_to_date(pm: int) -> date:
    day, _, _ = pm_to_clock(pm)
    return working_day_date(day)


def date_to_pm(iso_date: str) -> int:
    """Convert an export deadline to a productive-minute deadline (end of that calendar day)."""
    return date_to_day_offset(iso_date) * PPD + PPD


def fmt_date(day_offset: int) -> str:
    d = working_day_date(day_offset)
    return f"{d.strftime('%d/%m/%Y')} ({JOURS_FR[d.weekday()]})"
"""
utils/time_utils.py — Productive-minute time model
===================================================
Work hours: 08h00 to 00h00 (midnight) — 16 hours/day, no lunch break.

ENCODING — "productive minutes" (PM):
  PM 0    = 08h00  day 0
  PM 959  = 23h59  day 0
  PM 960  = 08h00  day 1
  ...

Constants:
  PPD = 960   productive minutes per day  (16h x 60)
"""

from datetime import date, timedelta

PPD            = 960   # 16 hours x 60 minutes
DAY_START_HOUR = 8
DAY_HOURS      = 16

# Legacy aliases kept for backward compatibility
MORN_MINS       = PPD
AFTN_MINS       = 0
LUNCH_START     = PPD
WORK_MINS_PER_DAY = PPD

START_DATE = date.today()
JOURS_FR   = ["Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim"]


# ---------------------------------------------------------------------------
# Calendar helpers
# ---------------------------------------------------------------------------

def working_day_date(offset_days: int) -> date:
    """Return the calendar date of working day `offset_days` from START_DATE (weekends skipped)."""
    if offset_days <= 0:
        return START_DATE
    d, n = START_DATE, 0
    while n < offset_days:
        d += timedelta(days=1)
        if d.weekday() < 5:
            n += 1
    return d


def date_to_day_offset(iso_date: str) -> int:
    """Convert an ISO date string to a working-day offset from START_DATE."""
    target = date.fromisoformat(iso_date)
    if target <= START_DATE:
        return 0
    d, count = START_DATE, 0
    while d < target:
        d += timedelta(days=1)
        if d.weekday() < 5:
            count += 1
    return count


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
      pm=0   -> (0,  8,  0)   08h00 day 0
      pm=959 -> (0, 23, 59)   23h59 day 0
      pm=960 -> (1,  8,  0)   08h00 day 1
    """
    day = pm // PPD
    off = pm % PPD
    h   = DAY_START_HOUR + off // 60
    m   = off % 60
    if h >= 24:
        h -= 24
    return day, h, m


def pm_to_hhmm(pm: int) -> str:
    _, h, m = pm_to_clock(pm)
    return f"{h:02d}h{m:02d}"


def pm_to_date(pm: int) -> date:
    day, _, _ = pm_to_clock(pm)
    return working_day_date(day)


def date_to_pm(iso_date: str) -> int:
    """Convert an export deadline to a productive-minute deadline (end of that working day)."""
    return date_to_day_offset(iso_date) * PPD + PPD


def fmt_date(day_offset: int) -> str:
    d = working_day_date(day_offset)
    return f"{d.strftime('%d/%m/%Y')} ({JOURS_FR[d.weekday()]})"
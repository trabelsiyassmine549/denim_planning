from datetime import date, timedelta

PPD            = 1440   # 24 hours x 60 minutes
DAY_START_HOUR = 0
DAY_HOURS      = 24


MORN_MINS         = PPD
AFTN_MINS         = 0
LUNCH_START       = PPD
WORK_MINS_PER_DAY = PPD

JOURS_FR = ["Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim"]


class _TodayProxy:
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

def working_day_date(offset_days: int) -> date:
    
    return date.today() + timedelta(days=max(offset_days, 0))


def date_to_day_offset(iso_date: str) -> int:
    
    target = date.fromisoformat(iso_date)
    delta  = (target - date.today()).days
    return max(delta, 1)


def date_to_offset(iso_date: str) -> int:
    return date_to_day_offset(iso_date)


def pm_to_clock(pm: int) -> tuple:
   
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
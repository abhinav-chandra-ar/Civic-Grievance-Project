import os
os.environ["DJANGO_SETTINGS_MODULE"] = "grievance_core.settings.dev"
import django
django.setup()

# ── Check 1: Audit endpoint permissions ──────────────────────────────────────
print("=== Audit endpoint permissions ===")
try:
    import importlib
    audit_views = importlib.import_module("apps.audit.views")
    for name in dir(audit_views):
        if name.startswith("_"):
            continue
        obj = getattr(audit_views, name)
        if hasattr(obj, "permission_classes"):
            print(f"  {name}.permission_classes = {obj.permission_classes}")
        if hasattr(obj, "queryset"):
            print(f"  {name}.queryset = {obj.queryset}")
except Exception as e:
    print("Error checking audit views:", e)

# ── Check 2: WorkflowEvent actor FK on submission ────────────────────────────
print("\n=== WorkflowEvent actor on fresh grievance ===")
from apps.grievances.services import submit_grievance
from apps.users.models import User
from apps.workflows.models import WorkflowEvent
import django.db.transaction as t

User.objects.filter(username="inv_p9_check").delete()
u = User.objects.create_user(username="inv_p9_check", password="X", role="citizen", email="inv@test.example")

sp = t.savepoint()
try:
    g = submit_grievance(submitter=u, raw_text="FK check test complaint for investigation")
    events = WorkflowEvent.objects.filter(grievance=g).order_by("id")
    print(f"Total workflow events: {events.count()}")
    for ev in events:
        print(f"  event #{ev.pk}: type={ev.transition_type!r}, actor_id={ev.actor_id!r}")
    first = events.first()
    if first:
        print(f"\nFirst event actor_id: {first.actor_id!r}")
        print(f"Citizen pk:          {u.pk!r}")
        print(f"Match: {first.actor_id == u.pk}")
finally:
    t.savepoint_rollback(sp)

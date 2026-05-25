"""Add image_file FileField to Attachment.

2026-05-25 — real binary storage for direct-upload path.
The FileField stores image bytes via Django's FileSystemStorage
(MEDIA_ROOT/grievance_attachments/YYYY/MM/DD/).

External-storage registrations continue to use the existing
storage_reference CharField and leave image_file NULL.
"""
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("attachments", "0003_initial"),
    ]

    operations = [
        migrations.AddField(
            model_name="attachment",
            name="image_file",
            field=models.FileField(
                blank=True,
                null=True,
                max_length=512,
                upload_to="grievance_attachments/%Y/%m/%d/",
                help_text=(
                    "Actual image binary stored via Django FileStorage.  "
                    "Populated only on direct-upload via POST /api/v1/attachments/upload/.  "
                    "External-storage registrations leave this blank."
                ),
            ),
        ),
    ]

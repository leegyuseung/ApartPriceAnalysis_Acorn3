from django.db import models

class Test(models.Model):
    num = models.IntegerField(primary_key=True)
    gu = models.CharField(max_length=45, blank=True, null=True)
    dong = models.CharField(max_length=45, blank=True, null=True)
    apart = models.CharField(max_length=45, blank=True, null=True)
    code = models.IntegerField(blank=True, null=True)
    price = models.IntegerField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'test'
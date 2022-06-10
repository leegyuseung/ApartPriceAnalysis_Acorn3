from django.db import models

# Create your models here.
class Users(models.Model):
    id = models.CharField(primary_key=True, max_length=20)
    name = models.CharField(max_length=45, blank=True, null=True)
    email = models.CharField(max_length=45, blank=True, null=True)
    addr = models.CharField(max_length=45, blank=True, null=True)
    phone = models.CharField(max_length=45, blank=True, null=True)
    pw = models.CharField(max_length=45, blank=True, null=True)
    
    class Meta:
        managed = False
        db_table = 'users'
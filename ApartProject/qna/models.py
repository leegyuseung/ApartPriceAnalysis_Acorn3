from django.db import models
from users.models import Users

# Create your models here.

class BoardTab(models.Model):

    userId = models.ForeignKey(Users, on_delete = models.CASCADE, related_name='user')
    title = models.CharField(max_length = 100)
    cont = models.TextField()
    bip = models.GenericIPAddressField()
    bdate = models.DateTimeField()
    readcnt = models.IntegerField()
    gnum = models.IntegerField()
    onum = models.IntegerField()
    nested = models.IntegerField()
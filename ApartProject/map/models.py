from django.db import models

class Test(models.Model):
    num = models.IntegerField(primary_key=True)
    gu = models.CharField(max_length=45, blank=True, null=True)
    dong = models.CharField(max_length=45, blank=True, null=True)
    apart = models.CharField(max_length=45, blank=True, null=True)
    code = models.IntegerField(blank=True, null=True)
    price = models.IntegerField(blank=True, null=True)
    we = models.CharField(max_length=45, blank=True, null=True)
    gd = models.CharField(max_length=45, blank=True, null=True)
    juso = models.CharField(max_length=45, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'test'

class Addrdata(models.Model):
    num = models.IntegerField(primary_key=True)
    price = models.IntegerField(blank=True, null=True)
    year = models.IntegerField(blank=True, null=True)
    dong = models.CharField(max_length=50, blank=True, null=True)
    apt = models.CharField(max_length=50, blank=True, null=True)
    month = models.IntegerField(blank=True, null=True)
    area = models.IntegerField(blank=True, null=True)
    bunji = models.CharField(max_length=50, blank=True, null=True)
    addr = models.CharField(max_length=50, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'addrData'
        
class Addrapt(models.Model):
    num = models.IntegerField(primary_key=True)
    addr = models.CharField(max_length=50, blank=True, null=True)
    apt = models.CharField(max_length=50, blank=True, null=True)
    dong = models.CharField(max_length=50, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'addrapt'
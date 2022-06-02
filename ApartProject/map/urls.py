from django.urls import path
from map import views

urlpatterns = [
     path('apart', views.apart),
     path('index', views.cssTest),  
     
]
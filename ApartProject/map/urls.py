from django.urls import path
from map import views

urlpatterns = [
     path('apart', views.apart),  
     path('graph', views.importData),
]
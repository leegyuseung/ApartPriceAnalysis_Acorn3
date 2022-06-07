from django.urls import path
from map import views

urlpatterns = [
    path('apart', views.apart),  
    path('graph', views.importData),
    path('polygun', views.polygun),
    path('pred',views.pred),
    
]
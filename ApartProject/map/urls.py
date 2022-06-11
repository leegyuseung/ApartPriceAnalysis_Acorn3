from django.urls import path
from map import views

urlpatterns = [
    path('apart', views.apart),  
    path('graph', views.importData),
    path('polygon', views.polygon),
    path('Dpolygon', views.Dpolygon), 
    path('pred',views.pred),
    path('dongmaker', views.dongmaker),

]
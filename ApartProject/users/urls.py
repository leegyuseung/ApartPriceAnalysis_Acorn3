from django.urls import path
from users import views

urlpatterns = [ 
    path('signup', views.signUp), 
    path('signupok', views.signUpOk), 
    path('login', views.login),  
    path('loginok', views.loginOk), 
    path('logout', views.logout),  
    path('mypage', views.myPage), 
    path('su', views.su), 
    path('suok', views.suOk),  
    path('tal', views.tal),  
    path('pw', views.pwC),   
    path('pwchange', views.pwChange),  
]   
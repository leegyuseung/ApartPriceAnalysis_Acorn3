from django.urls import path
from qna import views


urlpatterns = [
    path('list', views.listFunc),
    path('insert', views.insertFunc),
    path('insertok', views.insertOkFunc),
    path('search', views.searchFunc),
    path('content', views.contentFunc),
    path('update', views.updateFunc),
    path('updateok', views.updateOkFunc),
    path('delete', views.deleteFunc),
    path('deleteok', views.deleteOkFunc),
    
]

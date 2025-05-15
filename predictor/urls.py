from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),  # URL for the homepage
    path('predict/', views.predict_pneumonia, name='predict_pneumonia'),  # URL for handling predictions
]
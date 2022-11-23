from django.db import models
from django.contrib.auth.models import User

# Create your models here.
class Dataset(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE )
    file = models.FileField(upload_to='datasets/')
    timestamp = models.DateTimeField(auto_now_add=False, auto_now=True)

    # def __str__(self) -> str:
    #     # return self.user.first


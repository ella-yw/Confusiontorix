from django.views.generic import TemplateView
from django.shortcuts import render

from Web_Predictor import pred

import keras, pandas as pd
from sklearn.model_selection import train_test_split as tts

from django import forms
class uploadForm(forms.Form): file = forms.FileField(widget = forms.ClearableFileInput(attrs={'class': 'inputfile inputfile-4', 'style': 'display:none'}))

from django.db import models
class imageModel(models.Model): model_pic = models.ImageField(upload_to = '')

class home(TemplateView):
    def get(self, request): return render(request, 'home/index.html', {'form': uploadForm()})
    def post(self, request):
        if request.method == 'POST':
            form = uploadForm(request.POST, request.FILES)
            if form.is_valid():
                imageModel(model_pic = form.cleaned_data['file']).save()
                output, percentage = pred(str(form.cleaned_data['file'].name), keras, pd, tts)
                return render(request, 'home/results.html', {'form': form, 'output': output, 'percentage': percentage})
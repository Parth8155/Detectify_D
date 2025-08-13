from django import forms
from django.core.validators import FileExtensionValidator
from .models import Case, SuspectImage, VideoUpload


class CaseForm(forms.ModelForm):
    """Form for creating and editing cases"""
    
    class Meta:
        model = Case
        fields = ['name', 'description']
        widgets = {
            'name': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'Enter case name',
                'required': True
            }),
            'description': forms.Textarea(attrs={
                'class': 'form-control',
                'rows': 4,
                'placeholder': 'Enter case description (optional)'
            })
        }
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['name'].help_text = 'Give your case a descriptive name'
        self.fields['description'].help_text = 'Optional: Add details about the case'


class SuspectImageForm(forms.Form):
    """Form for uploading suspect images"""
    
    image = forms.FileField(
        widget=forms.FileInput(attrs={
            'class': 'form-control',
            'accept': 'image/png,image/jpeg,image/jpg',
            'required': True
        }),
        help_text='Upload a clear image of the suspect\'s face (PNG, JPG, JPEG)',
        validators=[FileExtensionValidator(allowed_extensions=['png', 'jpg', 'jpeg'])]
    )
    
    def clean_image(self):
        image = self.cleaned_data.get('image')
        
        if image:
            # Check file size (max 50MB for images)
            max_size = 50 * 1024 * 1024  # 50MB
            if image.size > max_size:
                raise forms.ValidationError('Image file is too large. Maximum size is 50MB.')
            
            # Check if it's actually an image
            try:
                from PIL import Image
                img = Image.open(image)
                img.verify()
            except Exception:
                raise forms.ValidationError('Invalid image file. Please upload a valid image.')
        
        return image


class VideoUploadForm(forms.Form):
    """Form for uploading videos"""
    
    video_file = forms.FileField(
        widget=forms.FileInput(attrs={
            'class': 'form-control',
            'accept': 'video/mp4,video/avi,video/mkv,video/mov',
            'required': True,
            'data-max-size': '500MB'
        }),
        help_text='Upload a video file to analyze (MP4, AVI, MKV, MOV - max 500MB). Progress will be shown during upload.',
        validators=[FileExtensionValidator(allowed_extensions=['mp4', 'avi', 'mkv', 'mov'])]
    )
    
    def clean_video_file(self):
        video = self.cleaned_data.get('video_file')
        
        if video:
            # Check file size (max 500MB)
            max_size = 500 * 1024 * 1024  # 500MB
            if video.size > max_size:
                raise forms.ValidationError(f'Video file is too large ({video.size / (1024*1024):.1f}MB). Maximum size is 500MB.')
            
            # Check minimum file size (1MB)
            min_size = 1 * 1024 * 1024  # 1MB
            if video.size < min_size:
                raise forms.ValidationError('Video file is too small. Minimum size is 1MB.')
        
        return video


class SearchForm(forms.Form):
    """Form for searching cases"""
    
    query = forms.CharField(
        max_length=200,
        required=False,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Search cases...',
        })
    )
    
    status = forms.ChoiceField(
        choices=[('', 'All Statuses')] + Case.STATUS_CHOICES,
        required=False,
        widget=forms.Select(attrs={
            'class': 'form-select'
        })
    )


class ProcessingOptionsForm(forms.Form):
    """Form for video processing options"""
    
    confidence_threshold = forms.FloatField(
        min_value=0.1,
        max_value=1.0,
        initial=0.85,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'step': '0.05',
            'min': '0.1',
            'max': '1.0'
        }),
        help_text='Confidence threshold for face matching (0.1 - 1.0)'
    )
    
    frame_interval = forms.FloatField(
        min_value=0.1,
        max_value=5.0,
        initial=0.25,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'step': '0.1',
            'min': '0.1',
            'max': '5.0'
        }),
        help_text='Frame sampling interval in seconds (0.1 - 5.0)'
    )
    
    buffer_before = forms.IntegerField(
        min_value=0,
        max_value=10,
        initial=2,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'min': '0',
            'max': '10'
        }),
        help_text='Seconds to include before detection in summary video'
    )
    
    buffer_after = forms.IntegerField(
        min_value=0,
        max_value=10,
        initial=3,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'min': '0',
            'max': '10'
        }),
        help_text='Seconds to include after detection in summary video'
    )

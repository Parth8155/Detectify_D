from django.shortcuts import render, get_object_or_404, redirect
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import JsonResponse, HttpResponse, Http404
from django.views.decorators.http import require_http_methods
from django.core.paginator import Paginator
from django.db.models import Q, Count
from django.conf import settings
from django.utils import timezone
import json

from .models import Case, SuspectImage, VideoUpload, DetectionResult, ProcessedVideo
from .forms import CaseForm, SuspectImageForm, VideoUploadForm

# Import streaming views
from ..media_processing.streaming_views import stream_stats, stop_stream


# --- REVERT TO SYNC VIEWS ---
# All views are now regular def functions, compatible with Django's synchronous template rendering

def dashboard(request):
    """Main dashboard view showing user's cases"""
    cases = Case.objects.filter(user=request.user).annotate(
        suspect_count=Count('suspect_images'),
        video_count=Count('videos'),
        detection_count=Count('videos__detections')
    )
    
    # Filter by status if requested
    status_filter = request.GET.get('status')
    if status_filter:
        cases = cases.filter(status=status_filter)
    
    # Search functionality
    search_query = request.GET.get('search')
    if search_query:
        cases = cases.filter(
            Q(name__icontains=search_query) | 
            Q(description__icontains=search_query)
        )
    
    # Pagination
    paginator = Paginator(cases, 10)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    
    context = {
        'cases': page_obj,
        'status_filter': status_filter,
        'search_query': search_query,
    }
    
    return render(request, 'cases/dashboard.html', context)


def case_detail(request, case_id):
    """Detailed view of a specific case"""
    case = get_object_or_404(Case, id=case_id, user=request.user)
    
    suspect_images = case.suspect_images.all()
    videos = case.videos.all()
    processed_videos = case.processed_videos.all()
    
    # Get detection statistics
    total_detections = DetectionResult.objects.filter(video__case=case).count()
    
    context = {
        'case': case,
        'suspect_images': suspect_images,
        'videos': videos,
        'processed_videos': processed_videos,
        'total_detections': total_detections,
    }
    
    return render(request, 'cases/case_detail.html', context)


def create_case(request):
    """Create a new case"""
    if request.method == 'POST':
        form = CaseForm(request.POST)
        if form.is_valid():
            case = form.save(commit=False)
            case.user = request.user
            case.save()
            messages.success(request, f'Case "{case.name}" created successfully!')
            return redirect('cases:case_detail', case_id=case.id)
    else:
        form = CaseForm()
    
    return render(request, 'cases/create_case.html', {'form': form})


def upload_suspect(request, case_id):
    """Upload suspect image to a case"""
    case = get_object_or_404(Case, id=case_id, user=request.user)
    
    if request.method == 'POST':
        form = SuspectImageForm(request.POST, request.FILES)
        if form.is_valid():
            try:
                # Create suspect instance without saving
                suspect = SuspectImage(case=case)
                
                # Save image data to database
                image_file = request.FILES['image']
                suspect.save_image_from_file(image_file)
                
                # Don't process immediately - just upload
                success_message = 'Suspect image uploaded successfully! Click "Process" to start face analysis.'
                
                # For AJAX requests, return JSON response
                if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                    return JsonResponse({
                        'status': 'success',
                        'message': success_message,
                        'suspect_id': suspect.id,
                        'redirect_url': f'/cases/{case.id}/'
                    })
                
                messages.success(request, success_message)
                return redirect('cases:case_detail', case_id=case.id)
            
            except Exception as e:
                error_message = f'Error uploading suspect image: {str(e)}'
                if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                    return JsonResponse({
                        'status': 'error',
                        'message': error_message
                    }, status=400)
                messages.error(request, error_message)
        else:
            # Form validation errors
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return JsonResponse({
                    'status': 'error',
                    'message': 'Invalid file. Please check file format and size requirements.'
                }, status=400)
    else:
        form = SuspectImageForm()
    
    context = {
        'form': form,
        'case': case,
    }
    
    return render(request, 'cases/upload_suspect.html', context)


def upload_video(request, case_id):
    """Upload video to a case"""
    case = get_object_or_404(Case, id=case_id, user=request.user)
    
    if request.method == 'POST':
        form = VideoUploadForm(request.POST, request.FILES)
        if form.is_valid():
            # Check for duplicate uploads (same filename, size, and recent timestamp)
            video_file = form.cleaned_data['video_file']
            recent_uploads = VideoUpload.objects.filter(
                case=case,
                video_name=video_file.name,
                uploaded_at__gte=timezone.now() - timezone.timedelta(minutes=5)
            )
            
            if recent_uploads.exists():
                messages.warning(request, 'This video appears to have been uploaded recently. Please check your case details.')
                return redirect('cases:case_detail', case_id=case.id)
            
            try:
                # Create video instance without saving
                video = VideoUpload(case=case)
                
                # Save video data to database
                video.save_video_from_file(video_file)
                
                # Don't extract metadata immediately - just upload
                success_message = 'Video uploaded successfully! Processing can be started manually.'
                
                # For AJAX requests, return JSON response
                if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                    return JsonResponse({
                        'status': 'success',
                        'message': success_message,
                        'video_id': video.id,
                        'redirect_url': f'/cases/{case.id}/'
                    })
                
                return redirect('cases:case_detail', case_id=case.id)
            
            except Exception as e:
                messages.error(request, f'Error uploading video: {str(e)}')
                if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                    return JsonResponse({
                        'status': 'error',
                        'message': f'Error uploading video: {str(e)}'
                    }, status=400)
        else:
            # Form validation errors
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return JsonResponse({
                    'status': 'error',
                    'message': 'Invalid file. Please check file format and size requirements.'
                }, status=400)
    else:
        form = VideoUploadForm()
    
    context = {
        'form': form,
        'case': case,
    }
    
    return render(request, 'cases/upload_video.html', context)


@login_required
@require_http_methods(["POST"])
def process_suspect(request, case_id, suspect_id):
    """Start processing a suspect image"""
    print("Processing suspect")
    case = get_object_or_404(Case, id=case_id, user=request.user)
    suspect = get_object_or_404(SuspectImage, id=suspect_id, case=case)
    
    if suspect.processed:
        return JsonResponse({
            'status': 'error',
            'message': 'Suspect image has already been processed.'
        })
    
    # Start processing
    from apps.media_processing.tasks import process_suspect_image_task
    try:
        print("process_suspect_image_task")
        # Try to run as async task
        task = process_suspect_image_task(suspect.id)
        task_id = 'sync_processing'
        message = 'Suspect image processing started!'
    except Exception as e:
        # If Celery fails, run synchronously
        try:        
            print("retry process_suspect_image_task")

            result = process_suspect_image_task(suspect.id)
            task_id = 'sync_processing'
            message = 'Suspect image processing completed!'
        except Exception as proc_error:
            return JsonResponse({
                'status': 'error',
                'message': f'Suspect processing failed: {str(proc_error)}'
            })
    
    return JsonResponse({
        'status': 'success',
        'message': message,
        'task_id': task_id
    })


@login_required
@require_http_methods(["POST"])
def extract_video_metadata(request, case_id, video_id):
    """Start extracting video metadata"""
    case = get_object_or_404(Case, id=case_id, user=request.user)
    video = get_object_or_404(VideoUpload, id=video_id, case=case)
    
    if video.duration:  # Already has metadata
        return JsonResponse({
            'status': 'error',
            'message': 'Video metadata has already been extracted.'
        })
    
    # Start metadata extraction
    from apps.media_processing.tasks import extract_video_metadata_task
    try:
        # Try to run as async task
        task = extract_video_metadata_task.delay(video.id)
        task_id = task.id
        message = 'Video metadata extraction started!'
    except Exception as e:
        # If Celery fails, run synchronously
        try:
            result = extract_video_metadata_task(video.id)
            task_id = 'sync_processing'
            message = 'Video metadata extraction completed!'
        except Exception as proc_error:
            return JsonResponse({
                'status': 'error',
                'message': f'Metadata extraction failed: {str(proc_error)}'
            })
    
    return JsonResponse({
        'status': 'success',
        'message': message,
        'task_id': task_id
    })


@login_required
@require_http_methods(["POST"])
def process_video(request, case_id, video_id):
    print("DEBUG: process_video URL was called (sync)")
    case = get_object_or_404(Case, id=case_id, user=request.user)
    print(f"Case found: {case}")
    video = get_object_or_404(VideoUpload, id=video_id, case=case)
    print(f"Video found: {video}")
    
    # Get suspects for this case
    suspects = case.suspect_images.filter(processed=True)
    
    if not suspects.exists():
        print("suspects not exist")
        return JsonResponse({
            'status': 'error',
            'message': 'No processed suspects found. Please upload and process suspect images first.'
        })
    
    if video.processed:
        print("video already processed")
        return JsonResponse({
            'status': 'error',
            'message': 'Video has already been processed.'
        })
    
    # Start processing
    suspect_ids = list(suspects.values_list('id', flat=True))

    # Set processing_started_at immediately so UI can reflect status
    video.processing_started_at = timezone.now()
    video.save()

    from apps.media_processing.tasks import process_video_task
    try:
        # Try to run as async task if possible
        if hasattr(process_video_task, 'delay'):
            task = process_video_task.delay(video.id, suspect_ids)
            task_id = getattr(task, 'id', 'celery')
            message = 'Video processing started!'
        else:
            result = process_video_task(video.id, suspect_ids)
            task_id = 'sync_processing'
            message = 'Video processing completed synchronously!'
    except Exception as proc_error:
        return JsonResponse({
            'status': 'error',
            'message': f'Video processing failed: {str(proc_error)}'
        })

    return JsonResponse({
        'status': 'success',
        'message': message,
        'task_id': task_id
    })
    


@login_required 
def video_processing_status(request, case_id, video_id):
    """AJAX endpoint to check video processing status"""
    case = get_object_or_404(Case, id=case_id, user=request.user)
    video = get_object_or_404(VideoUpload, id=video_id, case=case)
    
    # Calculate processing progress
    progress_percent = 0
    status = 'uploaded'
    message = 'Video uploaded successfully'
    current_frame = 0
    total_frames = 0
    frame_timestamp = 0
    detections_found = 0
    
    if video.processing_started_at:
        status = 'processing'
        message = 'Processing video for face detection...'
        
        # Try to get Celery task progress if available
        task_id = getattr(video, 'task_id', None)
        if task_id:
            try:
                from celery.result import AsyncResult
                result = AsyncResult(task_id)
                
                if result.state == 'PROGRESS':
                    meta = result.info
                    progress_percent = meta.get('progress', 0)
                    current_frame = meta.get('current_frame', 0)
                    total_frames = meta.get('total_frames', 0)
                    frame_timestamp = meta.get('timestamp', 0)
                    detections_found = meta.get('detections_found', 0)
                    
                    message = f'Processing frame {current_frame}/{total_frames} (Time: {frame_timestamp:.1f}s) - {detections_found} detections found'
                elif result.state == 'SUCCESS':
                    progress_percent = 100
                    status = 'completed'
                    message = 'Processing completed!'
                elif result.state == 'FAILURE':
                    status = 'error'
                    message = 'Processing failed'
            except Exception as e:
                print(f"Error getting task progress: {e}")
        
        # Estimate progress based on time elapsed (fallback)
        if progress_percent == 0 and video.duration:
            processing_time = (timezone.now() - video.processing_started_at).total_seconds()
            # Rough estimate: 1 second of processing per 10 seconds of video
            estimated_total_time = video.duration / 10
            progress_percent = min(90, (processing_time / estimated_total_time) * 100)
    
    if video.processed:
        status = 'completed'
        message = 'Video processing completed'
        progress_percent = 100
        
        # Get detection count
    detection_count = DetectionResult.objects.filter(video=video).count()
    # Consider results ready when there are detections persisted or a processed summary exists
    results_ready = detection_count > 0 or ProcessedVideo.objects.filter(original_video=video).exists()
    message = f'Processing completed! Found {detection_count} detections.'
    
    return JsonResponse({
        'status': status,
        'message': message,
        'progress': round(progress_percent, 1),
        'current_frame': current_frame,
        'total_frames': total_frames,
        'frame_timestamp': round(frame_timestamp, 1),
        'detections_found': detections_found,
    'duration': video.duration,
    'processing_started': video.processing_started_at.isoformat() if video.processing_started_at else None,
    'processing_completed': video.processing_completed_at.isoformat() if video.processing_completed_at else None,
    # Extra fields to help the frontend know when results are persisted
    'detection_count': detection_count,
    'results_ready': results_ready if video.processed else False,
    })
@login_required
def video_results(request, case_id, video_id):
    """View detection results for a video"""
    case = get_object_or_404(Case, id=case_id, user=request.user)
    video = get_object_or_404(VideoUpload, id=video_id, case=case)
    
    detections = DetectionResult.objects.filter(video=video).select_related('suspect')
    
    # Get processed/summary videos for this video
    processed_videos = ProcessedVideo.objects.filter(original_video=video)
    
    # Group detections by suspect
    detections_by_suspect = {}
    for detection in detections:
        suspect_id = detection.suspect.id
        if suspect_id not in detections_by_suspect:
            detections_by_suspect[suspect_id] = {
                'suspect': detection.suspect,
                'detections': []
            }
        detections_by_suspect[suspect_id]['detections'].append(detection)
    
    context = {
        'case': case,
        'video': video,
        'detections_by_suspect': detections_by_suspect,
        'total_detections': detections.count(),
        'processed_videos': processed_videos,
    }
    
    return render(request, 'cases/video_results.html', context)


@login_required
def download_processed_video(request, case_id, processed_video_id):
    """Download processed/summary video"""
    case = get_object_or_404(Case, id=case_id, user=request.user)
    processed_video = get_object_or_404(ProcessedVideo, id=processed_video_id, case=case)
    
    # Serve video data from database
    response = HttpResponse(
        processed_video.get_processed_data(),
        content_type='video/mp4'
    )
    response['Content-Disposition'] = f'attachment; filename="{processed_video.processed_name}"'
    return response


@login_required
def serve_suspect_image(request, case_id, suspect_id):
    """Serve suspect image from database"""
    case = get_object_or_404(Case, id=case_id, user=request.user)
    suspect = get_object_or_404(SuspectImage, id=suspect_id, case=case)
    
    # Determine content type based on image type
    content_type = f'image/{suspect.image_type}'
    if suspect.image_type == 'jpg':
        content_type = 'image/jpeg'
    
    response = HttpResponse(
        suspect.get_image_data(),
        content_type=content_type
    )
    response['Content-Disposition'] = f'inline; filename="{suspect.image_name}"'
    return response


@login_required
def serve_video(request, case_id, video_id):
    """Serve video from database"""
    case = get_object_or_404(Case, id=case_id, user=request.user)
    video = get_object_or_404(VideoUpload, id=video_id, case=case)
    
    response = HttpResponse(
        video.get_video_data(),
        content_type=f'video/{video.video_type}'
    )
    response['Content-Disposition'] = f'inline; filename="{video.video_name}"'
    return response


@login_required
def api_case_status(request, case_id):
    """API endpoint to get case processing status"""
    case = get_object_or_404(Case, id=case_id, user=request.user)
    
    # Get processing statistics
    total_videos = case.videos.count()
    processed_videos = case.videos.filter(processed=True).count()
    processing_videos = case.videos.filter(
        processing_started_at__isnull=False,
        processed=False
    ).count()
    
    total_suspects = case.suspect_images.count()
    processed_suspects = case.suspect_images.filter(processed=True).count()
    
    data = {
        'case_id': case.id,
        'status': case.status,
        'videos': {
            'total': total_videos,
            'processed': processed_videos,
            'processing': processing_videos,
        },
        'suspects': {
            'total': total_suspects,
            'processed': processed_suspects,
        },
        'updated_at': case.updated_at.isoformat(),
    }
    
    return JsonResponse(data)


@login_required
def delete_case(request, case_id):
    """Delete a case and all associated data"""
    case = get_object_or_404(Case, id=case_id, user=request.user)
    
    if request.method == 'POST':
        case_name = case.name
        case.delete()
        messages.success(request, f'Case "{case_name}" deleted successfully!')
        return redirect('cases:dashboard')
    
    context = {'case': case}
    return render(request, 'cases/delete_case.html', context)

@login_required
def people_detection_results(request, case_id, video_id):
    """View people detection results for a video"""
    case = get_object_or_404(Case, id=case_id, user=request.user)
    video = get_object_or_404(VideoUpload, id=video_id, case=case)
    # PeopleDetection model may not be available in some branches/environments
    try:
        from .models import PeopleDetection
        detections = PeopleDetection.objects.filter(video=video).order_by('timestamp')
    except Exception:
        detections = []

    # Calculate statistics (works for both QuerySet and list fallback)
    if hasattr(detections, 'count'):
        total_detections = detections.count()
        total_people_detected = sum(d.person_count for d in detections)
        max_people_in_frame = max([d.person_count for d in detections]) if total_detections else 0
    else:
        total_detections = len(detections)
        total_people_detected = sum(d.person_count for d in detections) if detections else 0
        max_people_in_frame = max([d.person_count for d in detections]) if detections else 0
    
    # Group detections by time intervals (e.g., every 10 seconds)
    timeline_data = []
    current_interval = 0
    interval_size = 10  # 10 seconds
    
    for detection in detections:
        interval = int(detection.timestamp // interval_size) * interval_size
        if interval != current_interval:
            current_interval = interval
            timeline_data.append({
                'start_time': interval,
                'end_time': interval + interval_size,
                'detections': []
            })
        
        if timeline_data:
            timeline_data[-1]['detections'].append(detection)
    
    context = {
        'case': case,
        'video': video,
        'detections': detections,
        'total_detections': total_detections,
        'total_people_detected': total_people_detected,
        'max_people_in_frame': max_people_in_frame,
        'timeline_data': timeline_data,
    }
    
    return render(request, 'cases/people_detection_results.html', context)

def suspect_image(request, suspect_id):
    from .models import SuspectImage
    import logging
    logger = logging.getLogger("django")
    try:
        suspect = SuspectImage.objects.get(id=suspect_id)
        if not suspect.image_data:
            logger.error(f"Suspect {suspect_id} has no image_data.")
            raise Http404()
        # Always use the correct MIME type for JPEG
        if suspect.image_type in ["jpg", "jpeg"]:
            content_type = "image/jpeg"
        elif suspect.image_type == "png":
            content_type = "image/png"
        else:
            content_type = "application/octet-stream"
        logger.info(f"Serving suspect image: id={suspect_id}, name={suspect.image_name}, type={suspect.image_type}, size={suspect.image_size}, content_type={content_type}")
        logger.info(f"First 16 bytes: {suspect.image_data[:16]}")
        return HttpResponse(suspect.image_data, content_type=content_type)
    except SuspectImage.DoesNotExist:
        logger.error(f"SuspectImage.DoesNotExist for id={suspect_id}")
        raise Http404()


@login_required
def stream_video_page(request, case_id, video_id):
    """Serve the real-time streaming page."""
    case = get_object_or_404(Case, id=case_id, user=request.user)
    video = get_object_or_404(VideoUpload, id=video_id, case=case)
    
    context = {
        'case': case,
        'video': video
    }
    
    return render(request, 'cases/stream_video.html', context)

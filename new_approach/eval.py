import os
import cv2
import json
import argparse
import numpy as np
from collections import defaultdict
from lip_detector import LipDetector

def load_ground_truth(ground_truth_path):
    """Load ground truth data from JSON file."""
    with open(ground_truth_path, 'r') as f:
        return json.load(f)

def calculate_metrics(ground_truth, predictions):
    """Calculate evaluation metrics comparing ground truth to predictions."""
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    true_negatives = 0
    total_comparisons = 0
    evaluated_faces = 0
    
    for frame_idx in ground_truth:
        if frame_idx not in predictions:
            continue
            
        gt_frame = ground_truth[frame_idx]
        pred_frame = predictions[frame_idx]
        
        # Get all unique face IDs from both ground truth and predictions
        all_faces = set(gt_frame.keys()) | set(pred_frame.keys())
        evaluated_faces += len(all_faces)
        
        for face_id in all_faces:
            # If face exists in both ground truth and predictions
            if face_id in gt_frame and face_id in pred_frame:
                gt_speaking = gt_frame[face_id]
                pred_speaking = pred_frame[face_id]
                
                if gt_speaking and pred_speaking:
                    true_positives += 1
                elif gt_speaking and not pred_speaking:
                    false_negatives += 1
                elif not gt_speaking and pred_speaking:
                    false_positives += 1
                else:  # not gt_speaking and not pred_speaking
                    true_negatives += 1
                    
                total_comparisons += 1
            # If face exists only in ground truth
            elif face_id in gt_frame:
                gt_speaking = gt_frame[face_id]
                if gt_speaking:
                    false_negatives += 1
                else:
                    true_negatives += 1
                total_comparisons += 1
            # If face exists only in predictions
            else:  # face_id in pred_frame
                pred_speaking = pred_frame[face_id]
                if pred_speaking:
                    false_positives += 1
                else:
                    true_negatives += 1
                total_comparisons += 1
    
    # Calculate metrics
    metrics = {}
    metrics["evaluated_frames"] = len(ground_truth)
    metrics["evaluated_faces"] = evaluated_faces
    metrics["total_comparisons"] = total_comparisons
    
    # Handle case where there are no valid comparisons
    if total_comparisons == 0:
        metrics["accuracy"] = 0
        metrics["precision"] = 0
        metrics["recall"] = 0
        metrics["f1_score"] = 0
        metrics["confusion_matrix"] = [[0, 0], [0, 0]]
        return metrics
    
    # Calculate accuracy
    metrics["accuracy"] = (true_positives + true_negatives) / total_comparisons if total_comparisons > 0 else 0
    
    # Calculate precision
    metrics["precision"] = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    
    # Calculate recall
    metrics["recall"] = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    
    # Calculate F1 score
    if metrics["precision"] + metrics["recall"] > 0:
        metrics["f1_score"] = 2 * (metrics["precision"] * metrics["recall"]) / (metrics["precision"] + metrics["recall"])
    else:
        metrics["f1_score"] = 0
    
    # Create confusion matrix
    metrics["confusion_matrix"] = [
        [true_negatives, false_positives],
        [false_negatives, true_positives]
    ]
    
    return metrics

def generate_predictions(video_path, sample_rate=30):
    """Generate predictions for the video using the lip detector."""
    predictions = {}
    lip_detector = LipDetector()
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process every Nth frame
        if frame_count % sample_rate == 0:
            # Detect faces and speaking status
            faces, speaking_status = lip_detector.detect_speaking(frame)
            
            # Store predictions
            frame_predictions = {}
            for i, (face, is_speaking) in enumerate(zip(faces, speaking_status)):
                frame_predictions[str(i)] = bool(is_speaking)
            
            predictions[str(frame_count)] = frame_predictions
        
        frame_count += 1
    
    cap.release()
    return predictions

def visualize_results(video_path, ground_truth, predictions, output_path, sample_rate=30):
    """Create a visualization video showing ground truth vs predictions."""
    lip_detector = LipDetector()
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process every Nth frame
        if frame_count % sample_rate == 0:
            frame_idx = str(frame_count)
            
            # Detect faces and speaking status
            faces, speaking_statuses = lip_detector.detect_speaking(frame)
            
            # Draw frame number
            cv2.putText(frame, f"Frame: {frame_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Draw ground truth and predictions
            if frame_idx in ground_truth and frame_idx in predictions:
                gt_frame = ground_truth[frame_idx]
                pred_frame = predictions[frame_idx]
                
                # Draw face information directly from lip_detector results
                for i, (face_obj, is_speaking) in enumerate(zip(faces, speaking_statuses)):
                    face_id = str(i)
                    
                    # Get the bounding box from the face object
                    # This depends on how your LipDetector returns faces
                    try:
                        # Try to get bbox from face_obj
                        if hasattr(face_obj, 'bbox'):
                            bbox = face_obj.bbox
                            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])
                        elif isinstance(face_obj, dict) and 'bbox' in face_obj:
                            bbox = face_obj['bbox']
                            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])
                        elif hasattr(face_obj, 'det_score'):
                            # InsightFace detection format
                            bbox = face_obj.bbox
                            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                        else:
                            # Print face object type for debugging
                            print(f"Face object type: {type(face_obj)}")
                            print(f"Face object attributes: {dir(face_obj)}")
                            # Skip this face if we can't determine the bbox
                            continue
                    except Exception as e:
                        print(f"Error processing face: {e}")
                        continue
                    
                    # Draw face rectangle
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Get ground truth and prediction status
                    gt_speaking = gt_frame.get(face_id, False)
                    pred_speaking = pred_frame.get(face_id, False)
                    
                    # Draw status text
                    gt_text = "GT: Speaking" if gt_speaking else "GT: Silent"
                    pred_text = "Pred: Speaking" if pred_speaking else "Pred: Silent"
                    
                    cv2.putText(frame, f"Face {face_id}", (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.putText(frame, gt_text, (x1, y1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    cv2.putText(frame, pred_text, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # Write frame to output video
        out.write(frame)
        frame_count += 1
    
    cap.release()
    out.release()

def main():
    parser = argparse.ArgumentParser(description="Evaluate lip detection performance")
    parser.add_argument("--video", required=True, help="Path to the input video file")
    parser.add_argument("--ground-truth", required=True, help="Path to the ground truth JSON file")
    parser.add_argument("--output-dir", required=True, help="Directory to save evaluation results")
    parser.add_argument("--sample-rate", type=int, default=30, help="Sample rate for frame processing")
    parser.add_argument("--visualize", action="store_true", help="Generate visualization video")
    parser.add_argument("--predictions", help="Path to pre-generated predictions JSON file (optional)")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load ground truth data
    ground_truth = load_ground_truth(args.ground_truth)
    
    # Generate or load predictions
    if args.predictions:
        with open(args.predictions, 'r') as f:
            predictions = json.load(f)
    else:
        predictions = generate_predictions(args.video, args.sample_rate)
        
        # Save predictions to file
        predictions_path = os.path.join(args.output_dir, "predictions.json")
        with open(predictions_path, 'w') as f:
            json.dump(predictions, f, indent=4)
    
    # Calculate metrics
    metrics = calculate_metrics(ground_truth, predictions)
    
    # Save metrics to file
    metrics_path = os.path.join(args.output_dir, "evaluation_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # Print metrics
    print("\nEvaluation Metrics:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    print(f"Evaluated Frames: {metrics['evaluated_frames']}")
    print(f"Evaluated Faces: {metrics['evaluated_faces']}")
    print(f"Total Comparisons: {metrics['total_comparisons']}")
    print("\nConfusion Matrix:")
    print(f"TN: {metrics['confusion_matrix'][0][0]}, FP: {metrics['confusion_matrix'][0][1]}")
    print(f"FN: {metrics['confusion_matrix'][1][0]}, TP: {metrics['confusion_matrix'][1][1]}")
    
    # Generate visualization if requested
    if args.visualize:
        visualization_path = os.path.join(args.output_dir, "visualization.mp4")
        visualize_results(args.video, ground_truth, predictions, visualization_path, args.sample_rate)
        print(f"\nVisualization saved to: {visualization_path}")

if __name__ == "__main__":
    main() 
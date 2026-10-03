# Face Verification with OpenCV and DeepFace

This is a small computer vision project I built to practice working with face recognition models and real-time webcam input.

The program captures frames from a webcam and compares the detected face against previously provided reference images. It then returns a verification result based on the comparison.

## How it works

At a high level, the workflow is:

Webcam frame → face detection / processing → comparison with reference image → verification result

The project uses pretrained face-recognition functionality rather than training a facial-recognition model from scratch.

## Technologies

- Python
- OpenCV
- DeepFace

## Purpose

I built this project mainly as personal practice.

My goal was to understand how an existing computer vision model could be integrated into a small application and how changes to the comparison process affected whether a face was successfully recognised.

## Limitations

This is an experimental learning project and should not be treated as a production biometric system.

The current implementation has several limitations:

- Verification accuracy is not reliable enough for security-sensitive use.
- Results can depend on the quality of the reference image.
- Lighting, camera position, facial angle, and other image conditions can affect the result.
- The project has not been evaluated on a large or representative dataset.

Reference photographs are intentionally not included in the public repository.

## Running the project

The project is intended to run locally with a webcam and one or more reference images.

Exact installation and execution instructions will be added after the repository dependencies and entry point are verified.

## Future work

I may return to the project to improve the verification process, evaluate it more systematically, and make the handling of reference images and webcam input more robust.

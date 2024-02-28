# 자폐 아동을 위한 감정 학습 앱 개발 (FER)

</br>

## I. Intro

</br>

### Objective

자폐 스펙트럼 장애를 가진 아동들은 종종 감정을 이해하고 타인과의 사회적 상호작용에서 이를 적절히 표현하는 데 어려움을 겪는다. 자폐 아동의 감정 인식 및 표현 능력을 향상시키기 위해, 생성된 감정을 통해 학습하는 라이브러리를 개발한다. 자폐 아동이 다양한 감정을 인식하고, 자신의 감정을 적절히 표현하는 데 필요한 지원을 제공하는 것을 목적으로 한다. 

</br>

### Overview

이 프로젝트는 MMA Facial Expression 데이터셋을 사용하여 감정을 인식하는 딥러닝 모델을 개발하고, 이를 Android 앱에 통합하여 실시간으로 감정을 분석하는 기능을 구현한다. MobileNetV2 아키텍처를 기반으로 한 모델 학습 과정과 Android Studio를 사용한 앱 개발 과정이 이루어진다. 이를 통해 자폐아동의 표정을 구분하고 (행복, 슬픔, 화남, 혐오, 놀람) 피드백 메세지를 통해 적절한 학습을 할 수 있도록 도와준다. 

### 1. MMA Facial Expression Dataset

[MMAFEDB](https://www.kaggle.com/datasets/mahmoudima/mma-facial-expression/data)는 다양한 표정 데이터 세트에서 수집된다. 모든 이미지는 얼굴 영역만 잘리고 48x48픽셀로 크기가 조정된다. 표정은 총 7가지로 angry, disgust, fear, happy, neutral, sad, surprise 라벨링 되어있다. 그 중 생성모델에서 사용되는 감정인 surprise, sad, happy, fear, angry만 사용한다.

### 2. FER Model

[FER Model](https://www.kaggle.com/yunnajoo/fer-model/edit)으로 감정인식 모델을 생성한다. 안드로이드 스튜디오에서 사용하는 tflite 형식으로 저장한다. 추후에 Android Project File에서 assets 폴더를 생성하여 모델을 넣어둔다.

### 3. Android Studio

raw 폴더에 OpenCV에서 얼굴 검출을 위해 사용하는 Haar Cascade 분류기 구성 파일을 추가하고 OpenCV 라이브러리의 안드로이드 포팅 버전인 openCVLibrary3414 폴더를 추가하여 감정인식 모델이 잘 작동할 수 있도록 한다.  또한 ,**`jniLibs`** 폴더를 추가하여 네이티브 라이브러리들을 Java 코드와 통합할 수 있게 한다.

</br></br></br>

## II. Architecture Details

### EmotionRecognition.java : **얼굴 감정 인식(FER) 모델**

Android 애플리케이션 내의 **`Emodion Recognition`** 클래스는 TensorFlow Lite와 OpenCV를 사용하는 얼굴 감정 인식 컴포넌트로서의 역할을 한다. 이 클래스의 주요 작업은 기계 학습 모델을 로드하고, 이미지를 처리하여 얼굴을 검출하며, 검출된 얼굴 표정을 바탕으로 감정을 분류한다.

</br>

### Key Components

- **CascadeClassifier (cascadeClassifier)**: OpenCV의 Haar 특징 기반 캐스케이드 분류기를 사용하여 이미지 내의 얼굴을 검출한다. **`haarcascade_frontalface_alt.xml`** 파일을 로드하여 얼굴 영역을 식별한다.
- **TensorFlow Lite Interpreter (interpreter)**: 이 인터프리터는 사전 훈련된 TensorFlow Lite 모델을 실행하여 잘라낸 얼굴 이미지로부터 감정을 분류한다. 모델은 **`loadModelFile`** 메소드를 사용하여 assets에서 로드되며, **`GpuDelegate`**에 의해 제공된 GPU 가속을 사용하여 실행된다.

</br>

### Image Handler

- OpenCV의 **`Imgproc.cvtColor`**을 사용하여 카메라 프레임을 회색조로 변환한다.
- **`cascadeClassifier.detectMultiScale`**을 통해 회색조 이미지 내의 얼굴을 검출한다.
- 얼굴 영역을 잘라내어 비트맵을 생성하고, TensorFlow Lite 모델이 예상하는 입력 크기로 크기를 조정한다.
- 비트맵을 TensorFlow Lite 모델의 입력으로 사용하기 위한 **`ByteBuffer byteBuffer`**로 변환한다.

</br>

### Emotion Prediction

- 인터프리터는 얼굴 이미지 데이터를 포함한 **`byteBuffer`**를 사용하여 감정 인식 모델을 실행한다.
- 모델 출력의 후처리에는 놀람 감정의 확신 값 정규화 및 결과를 문자열 **`emotion_s`**로 형식화하는 작업이 포함된다.

</br>

### Utility methods

- **`convertBitmapToByteBuffer`**: 얼굴의 비트맵을 TensorFlow Lite 모델의 입력 형식에 맞게 **`ByteBuffer`**로 변환합니다.
- **`loadModelFile`**: TensorFlow Lite 모델 파일을 에셋에서 읽고 메모리에 매핑하여 인터프리터가 사용할 수 있도록 합니다.

</br>

### Exception handling

- 생성자와 메소드는 파일 접근이나 모델 로딩 문제 발생 시 애플리케이션이 유지될 수 있도록 IO 예외를 관리하는 에러 핸들링을 포함한다.

</br></br></br>

## III. DataSets & Performance

</br>

### MMA FACIAL EXPRESSION

`label = ['surprise', 'sad', 'happy', 'fear', 'angry']` 

→ 생성모델의 label에 맞춰서 진행한다.

</br>

### MobileNet V2

→ OnDevice 구현 목적이기 때문에 경량화된 모델 사용한다.

</br></br></br>

## IV. Environment Set-up

### Download App File

Use git clone or directly download. Then you can get a below model hierarchy.

```
- app
  - manifests
    - AndroidManifest.xml   
  - java
    - com.example.imagepro
      - AngryActivity
      - AngryRecognition
      - FearActivity
      - FearRecognition
      - HappyActivity
      - HappyRecognition
      - MainActivity
      - SadActivity
      - SadRecognition
      - SurpriseActivity
      - SurpriseRecognition
  - java (generated)
  - assets     // Add Model File
    - model.tflite
    - model300.tflite
    - newmodel.tflite
  - jniLibs  // Add Folder
  - res
    - drawable
    - layout
      - activity_angry.xml
      - activity_fear.xml
      - activity_happy.xml
      - activity_main.xml
      - activity_sad.xml
      - activity_surprise.xml
    - mipmap
    - raw
      - haarcascade_frontalface_alt.xml  // Add File
    - values
  - res (generated)
- openCVLibrary3413 // Add Folder
- Gradle Scripts
```
</br></br></br>

## V. Miscellaneous

> Android Studio : APP Contents </br>
> 

사용자는 실시간으로 스트리밍되는 자신의 얼굴을 보며, 앱이 분석한 감정 상태(예: 놀람, 슬픔, 행복 등)를 확인할 수 있다. 각 감정 상태는 버튼으로 표시되며, 사용자가 이를 클릭하면 해당 감정에 대한 피드백을 받을 수 있다. 전체적인 프로세스는 사용자의 상호작용을 바탕으로 감정 인식의 정확도를 향상시키는 데 목적이 있다. 피드백 메세지를 통해 특정 임계값 이하이면 감정에 대한 개념 피드백을 제공하고, 임계값 이상이면 응원해주는 말로 격려 피드백을 제공한다.

</br>

<img width="878" alt="image" src="https://github.com/multi-sensor/FER_EmoRecognition/assets/54527982/86df6bc5-17f5-4532-a20d-3263252e16bc">

</br></br>

> 시연 영상

</br>

https://github.com/multi-sensor/FER_EmoRecognition/assets/54527982/6f774294-534b-42df-b21f-566d3b4e60e1

</br></br></br>

## VII. Requirements

```
Python 3.11.5
Android Studio Dolphin | 2021.03.01
CompileSdkVersion 29
BuildToolsVersion "30.0.2"
MinSdkVersion 21
TargetSdkVersion 29

* dependencies
implementation 'org.tensorflow:tensorflow-lite-metadata:0.1.0-rc1'
implementation 'org.tensorflow:tensorflow-lite-gpu:2.2.0'
implementation 'org.tensorflow:tensorflow-lite-support:0.1.0'
implementation 'org.tensorflow:tensorflow-lite-task-vision:0.1.0'
implementation 'org.tensorflow:tensorflow-lite-task-text:0.1.0'

implementation 'androidx.appcompat:appcompat:1.2.0'
implementation 'com.google.android.material:material:1.3.0'
implementation 'androidx.constraintlayout:constraintlayout:2.0.4'
implementation project(path: ':openCVLibrary3413')
testImplementation 'junit:junit:4.+'
androidTestImplementation 'androidx.test.ext:junit:1.1.2'
androidTestImplementation 'androidx.test.espresso:espresso-core:3.3.0'
```
</br></br></br>

## VIII. Reference
[MMA DATASET](https://www.kaggle.com/datasets/mahmoudima/mma-facial-expression) </br>
[MobileNetV2](https://arxiv.org/abs/1801.04381)</br>
[JavaCamera](https://docs.nvidia.com/gameworks/content/technologies/mobile/opencv_tutorial_camera_preview.htm)</br>



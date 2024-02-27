package com.example.imagepro;

import android.app.Activity;
import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.util.Log;
import android.widget.TextView;

import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.GpuDelegate;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;

public class AngryRecognition {
    private CascadeClassifier cascadeClassifier;
    private Interpreter interpreter;
    private int INPUT_SIZE;
    private int height=0;
    private int width=0;
    private GpuDelegate gpuDelegate=null;
    private TextView feedbackText;
    private Context context; // Context 멤버 변수 추가

    public AngryRecognition(AssetManager assetManager, Context context, String modelPath, int inputSize) throws IOException {
        INPUT_SIZE=inputSize;
        Interpreter.Options options=new Interpreter.Options();
        gpuDelegate=new GpuDelegate();
        options.addDelegate(gpuDelegate);
        options.setNumThreads(4); // set this according to your phone
        interpreter=new Interpreter(loadModelFile(assetManager,modelPath),options);
        Log.d("facial_Expression","Model is loaded");

        this.context = context; // 매개변수로 받은 Context를 멤버 변수에 할당
        this.feedbackText = feedbackText;

        // now we will load haarcascade classifier
        try {
            // define input stream to read classifier
            InputStream is=context.getResources().openRawResource(R.raw.haarcascade_frontalface_alt);
            // create a folder
            File cascadeDir=context.getDir("cascade",Context.MODE_PRIVATE);
            // now create a new file in that folder
            File mCascadeFile=new File(cascadeDir,"haarcascade_frontalface_alt");
            // now define output stream to transfer data to file we created
            FileOutputStream os=new FileOutputStream(mCascadeFile);
            // now create buffer to store byte
            byte[] buffer=new byte[4096];
            int byteRead;
            // read byte in while loop
            // when it read -1 that means no data to read
            while ((byteRead=is.read(buffer)) !=-1){
                // writing on mCascade file
                os.write(buffer,0,byteRead);

            }
            // close input and output stream
            is.close();
            os.close();
            cascadeClassifier=new CascadeClassifier(mCascadeFile.getAbsolutePath());
            // if cascade file is loaded print
            Log.d("facial_Expression","Classifier is loaded");

        }
        catch (IOException e){
            e.printStackTrace();
        }

    }

    // feedbackText TextView 설정을 위한 메서드
    public void setFeedbackTextView(TextView feedbackText) {
        this.feedbackText = feedbackText;
    }

    public Mat recognizeImage(Mat mat_image, int rotation) {
        Mat grayscaleImage = new Mat();
        Imgproc.cvtColor(mat_image, grayscaleImage, Imgproc.COLOR_RGBA2GRAY);

        MatOfRect faces = new MatOfRect();
        if (cascadeClassifier != null) {
            cascadeClassifier.detectMultiScale(grayscaleImage, faces, 1.1, 2, 2, new Size(), new Size());
        }

        Rect[] faceArray = faces.toArray();

        if (faceArray.length > 0) {
            Rect face = faceArray[0]; // 첫 번째 얼굴

            Mat faceROI = new Mat(grayscaleImage, face);
            Bitmap faceBitmap = Bitmap.createBitmap(faceROI.cols(), faceROI.rows(), Bitmap.Config.ARGB_8888);
            Utils.matToBitmap(faceROI, faceBitmap);
            Bitmap resizedBitmap = Bitmap.createScaledBitmap(faceBitmap, INPUT_SIZE, INPUT_SIZE, true);
            ByteBuffer byteBuffer = convertBitmapToByteBuffer(resizedBitmap);

            float[][] emotionPrediction = new float[1][1];
            interpreter.run(byteBuffer, emotionPrediction);

            float emotion_v = emotionPrediction[0][0];
            String emotion_s;
            float normalizedEmotionValue; // 정규화된 감정 값을 저장할 변수

            if (emotion_v >= 1.5 && emotion_v < 2.5) {
                float minValue = 1.5f; // 현재 값의 최소 범위
                float maxValue = 2.5f; // 현재 값의 최대 범위
                // (현재 값 - 최소 범위) / (최대 범위 - 최소 범위)를 통해 정규화
                normalizedEmotionValue = (emotion_v - minValue) / (maxValue - minValue);
                emotion_s = "Angry";
                //emotion_s = "Angry (" + String.format("%.1f", normalizedEmotionValue) + ")";
            }

            else {
                emotion_s = "Not Angry";
                normalizedEmotionValue = -1;

                final String emotionText = "Emotion Concept";

                // 메인 스레드에서 feedbackText 텍스트 뷰 업데이트
                if (context instanceof Activity) {
                    ((Activity)context).runOnUiThread(new Runnable() {
                        @Override
                        public void run() {
                            if (feedbackText != null) {
                                feedbackText.setText(emotionText); // UI 업데이트
                            }
                        }
                    });
                }
            }

            Point textPosition = new Point(face.x + 10, face.y + 20);
            // 스트로크 텍스트 (흰색)
            int textSize = 2; // 텍스트 크기
            int thickness = 8; // 텍스트 두께
            Imgproc.putText(mat_image, emotion_s, new Point(face.x, face.y - 10), Core.FONT_HERSHEY_SIMPLEX, textSize, new Scalar(255, 255, 255), thickness);

            // 실제 텍스트 (검은색)
            thickness = 2; // 실제 텍스트의 두께를 줄임
            Imgproc.putText(mat_image, emotion_s, new Point(face.x, face.y - 10), Core.FONT_HERSHEY_SIMPLEX, textSize, new Scalar(0, 0, 0), thickness);

            // 감정 값에 따라 텍스트 결정
            final String emotionText = determineEmotionText(emotion_v); // 이 부분이 감정 분석 결과를 처리

            // 메인 스레드에서 feedbackText 텍스트 뷰 업데이트
            if (context instanceof Activity) {
                ((Activity)context).runOnUiThread(new Runnable() {
                    @Override
                    public void run() {
                        if (feedbackText != null) {
                            feedbackText.setText(emotionText); // UI 업데이트
                        }
                    }
                });
            }

        }

        return mat_image;
    }

    private String determineEmotionText(float emotion_v) {
        if (emotion_v >= 0.0 && emotion_v < 1) {
            return "Good!";
        } else
            return "Emotion Concept";
    }



//    private String determineEmotionText(float emotion_v) {
//        if (emotion_v < 0.3) {
//            return "Emotion Concept";
//        } else if (emotion_v >= 0.3 && emotion_v < 0.5) {
//            return "GOOD";
//        } else {
//            return "Cheer UP";
//        }
//    }



    private ByteBuffer convertBitmapToByteBuffer(Bitmap scaledBitmap) {
        ByteBuffer byteBuffer;
        int size_image=INPUT_SIZE;//48

        byteBuffer=ByteBuffer.allocateDirect(4*1*size_image*size_image*3);
        byteBuffer.order(ByteOrder.nativeOrder());
        int[] intValues=new int[size_image*size_image];
        scaledBitmap.getPixels(intValues,0,scaledBitmap.getWidth(),0,0,scaledBitmap.getWidth(),scaledBitmap.getHeight());
        int pixel=0;
        for(int i =0;i<size_image;++i){
            for(int j=0;j<size_image;++j){
                final int val=intValues[pixel++];
                // now put float value to bytebuffer
                // scale image to convert image from 0-255 to 0-1
                byteBuffer.putFloat((((val>>16)&0xFF))/255.0f);
                byteBuffer.putFloat((((val>>8)&0xFF))/255.0f);
                byteBuffer.putFloat(((val & 0xFF))/255.0f);

            }
        }
        return byteBuffer;
    }

    private MappedByteBuffer loadModelFile(AssetManager assetManager, String modelPath) throws IOException{
        // this will give description of file
        AssetFileDescriptor assetFileDescriptor=assetManager.openFd(modelPath);
        // create a inputsteam to read file
        FileInputStream inputStream=new FileInputStream(assetFileDescriptor.getFileDescriptor());
        FileChannel fileChannel=inputStream.getChannel();

        long startOffset=assetFileDescriptor.getStartOffset();
        long declaredLength=assetFileDescriptor.getDeclaredLength();
        return  fileChannel.map(FileChannel.MapMode.READ_ONLY,startOffset,declaredLength);

    }
}
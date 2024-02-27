package com.example.imagepro;

import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import android.Manifest;
import android.app.Activity;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.os.Handler;
import android.util.Log;
import android.view.SurfaceView;
import android.view.View;
import android.view.Window;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.TextView;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;

import java.io.IOException;

public class HappyActivity extends Activity implements CameraBridgeViewBase.CvCameraViewListener2{
    private static final String TAG="MainActivity";
    private static final int MY_PERMISSIONS_REQUEST_CAMERA = 100;

    private Mat mRgba;
    private Mat mGray;
    private CameraBridgeViewBase mOpenCvCameraView;
    private HappyRecognition happyRecognition;
    private TextView feedbackText;

    private TextView emotionText;
    private Button nextButton;

    private Handler handler = new Handler(); // UI 업데이트를 위한 핸들러
    private String lastEmotion = ""; // 마지막으로 인식된 감정을 저장할 변수

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status){
                case LoaderCallbackInterface
                        .SUCCESS:{
                    Log.i(TAG,"OpenCv Is loaded");
                    mOpenCvCameraView.enableView();
                }
                default:
                {
                    super.onManagerConnected(status);
                }
                break;
            }
        }
    };

    public HappyActivity(){
        Log.i(TAG,"Instantiated new " + this.getClass());
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        requestWindowFeature(Window.FEATURE_NO_TITLE);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        setContentView(R.layout.activity_happy);

        nextButton = findViewById(R.id.nextbutton);
        feedbackText = findViewById(R.id.feedbackText);

        int MY_PERMISSIONS_REQUEST_CAMERA=0;
        // if camera permission is not given it will ask for it on device
        if (ContextCompat.checkSelfPermission(HappyActivity.this, Manifest.permission.CAMERA)
                == PackageManager.PERMISSION_DENIED){
            ActivityCompat.requestPermissions(HappyActivity.this, new String[] {Manifest.permission.CAMERA}, MY_PERMISSIONS_REQUEST_CAMERA);
        }

        mOpenCvCameraView=(CameraBridgeViewBase) findViewById(R.id.frame_Surface);
        mOpenCvCameraView.setCameraIndex(CameraBridgeViewBase.CAMERA_ID_FRONT);
        mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this);

        try{
            // input size of model is 48
            int inputSize=48;
            happyRecognition=new HappyRecognition(getAssets(), HappyActivity.this, "model300.tflite",inputSize);
            happyRecognition.setFeedbackTextView(feedbackText);
        }
        catch (IOException e){
            e.printStackTrace();
        }

        nextButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(HappyActivity.this, FearActivity.class);
                startActivity(intent);
            }
        });

    }

    // 감정에 따라 피드백 텍스트를 업데이트하는 메서드
    private void updateFeedbackText(final String emotion) {
        // UI 스레드에서 텍스트 뷰 업데이트
        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                feedbackText.setText(emotion); // 텍스트 뷰에 감정 표시
            }
        });

        // 기존에 예약된 작업이 있다면 취소 (텍스트가 사라지는 작업)
        handler.removeCallbacksAndMessages(null);

        // 3초 후에 텍스트 뷰의 내용을 지우는 작업 예약
        handler.postDelayed(new Runnable() {
            @Override
            public void run() {
                feedbackText.setText(""); // 3초 후 텍스트 뷰 비우기
            }
        }, 3000); // 3초 지연
    }


    @Override
    protected void onResume() {
        super.onResume();
        if (OpenCVLoader.initDebug()){
            //if load success
            Log.d(TAG,"Opencv initialization is done");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
        else{
            //if not loaded
            Log.d(TAG,"Opencv is not loaded. try again");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_4_0,this,mLoaderCallback);
        }
    }

    @Override
    protected void onPause() {
        super.onPause();
        if (mOpenCvCameraView !=null){
            mOpenCvCameraView.disableView();
        }
    }

    public void onDestroy(){
        super.onDestroy();
        if(mOpenCvCameraView !=null){
            mOpenCvCameraView.disableView();
        }

    }

    public void onCameraViewStarted(int width ,int height){
        mRgba=new Mat(height,width, CvType.CV_8UC4);
        mGray =new Mat(height,width,CvType.CV_8UC1);

        mOpenCvCameraView.setMaxFrameSize(640, 480);
    }

    public void onCameraViewStopped(){
        mRgba.release();
    }

    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame){
        mRgba = inputFrame.rgba();
        mGray = inputFrame.gray();

        // 감정 인식 및 기타 처리
        mRgba = happyRecognition.recognizeImage(mRgba,0);

        return mRgba;
    }

}
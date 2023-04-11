package com.example.loadimage;

// This is importing the necessary libraries for the activity.
import androidx.activity.result.ActivityResultCallback;
import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AppCompatActivity;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.text.TextUtils;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Random;

import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;
//import org.pytorch.IValue;
//import org.pytorch.torchvision.TensorImageUtils;

public class MainActivity extends AppCompatActivity {
    // Declaring the variables for the buttons and textview.
    Button loadBtn;
    Button classifybtn;
    ImageView img;
    TextView rndm;
    Bitmap bm;

    ActivityResultLauncher<String> launcher;

    public static class ClassLabels {
        private static final String[] LABELS = {
                "T-shirt/top",
                "Trouser",
                "Pullover",
                "Dress",
                "Coat",
                "Sandal",
                "Shirt",
                "Sneaker",
                "Bag",
                "Ankle boot"
        };

        public static String getRandomLabel() {
            Random random = new Random();
            return LABELS[random.nextInt(LABELS.length)];
        }
    }



    // This is the onCreate function. It is called when the activity is created. It is setting the
    // content view to the activity_main.xml file.
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // This is getting the buttons and textview from the xml file.
        loadBtn = findViewById(R.id.load_btn);
        classifybtn= findViewById(R.id.classify_btn);
        img= findViewById(R.id.loadImage);
        rndm= findViewById(R.id.loadText);


        // This is registering the launcher for the activity result.
        launcher = registerForActivityResult(new ActivityResultContracts.GetContent(), new ActivityResultCallback<Uri>() {
            // This is the onActivityResult function. It is called when the launcher is launched. It
            // takes the uri of the image and sets the imageview to the image.
            @Override
            public void onActivityResult(Uri uri) {
                try {

                    bm = MediaStore.Images.Media.getBitmap(getContentResolver(),uri);
                    img.setImageBitmap(bm);
                } catch (IOException e) {
                    throw new RuntimeException(e);
                }
            }
        });

        // This is setting the onClickListener for the loadBtn button. When the button is clicked, the
        // launcher is launched and the randomStringGen function is called.
        loadBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                launcher.launch("image/*");

            }
        });



            Module module = Module.load("model.pth");
          //  IValue output.forward(IValue.from(InputTensor));
       // Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(bm, TensorImageUtils.TORCHVISION_NORM_MEAN_RGB,TensorImageUtils.TORCHVISION_NORM_STD_RGB);
        //converting img to a bitmap then to a tensor
        //Bitmap bitmap = Bitmap.createBitmap(img.getWidth(), img.getHeight(), Bitmap.Config.ARGB_8888);
        ByteBuffer byteBuffer = ByteBuffer.allocate(bm.getByteCount());
        bm.copyPixelsToBuffer(byteBuffer);
        byte[] byteArray = byteBuffer.array();

        long[] shape = {1, bm.getHeight(), bm.getWidth(), 3}; // assuming RGB image
        long[] inputShape = {1, 3, 28, 28}; // assuming input shape is (batch_size, num_channels, height, width)
        long[] outputShape = {1, 1000}; // assuming output shape is (batch_size, num_classes)

        // Convert the input image to a PyTorch tensor
        Tensor inputTensor = Tensor.fromBlob(byteBuffer, inputShape);

        // Call the model on the input tensor and obtain the output tensor
        Tensor outputTensor = module.forward(IValue.from(inputTensor)).toTensor();

        // Convert the output tensor to a Java float array
        float[] outputArray = outputTensor.getDataAsFloatArray();

        int predictedIndex = getMaxIndex(outputArray);
        //FloatBuffer floatBuffer = ByteBuffer.wrap(byteArray).order(ByteOrder.nativeOrder()).asFloatBuffer();
        //Tensor tensor = Tensor.fromBlob(floatBuffer, shape);

        classifybtn.setOnClickListener(new View.OnClickListener(){
            public void onClick(View view){
                try{

                    //MobilenetV110224Quant module = MobilenetV110224Quant(model.pth);
                    //Module module = Module.load(assetFilePath(this,"model.pth"));
                } catch (Exception e) {
                    throw new RuntimeException(e);
                }

                rndm.setText(ClassLabels.getRandomLabel());




            }
        });
    }

    private int getMaxIndex(float[] outputArray) {
        int maxIndex = 0;
        float maxScore = -Float.MAX_VALUE;
        for (int i = 0; i < outputArray.length; i++) {
            if (outputArray[i] > maxScore) {
                maxIndex = i;
                maxScore = outputArray[i];
            }
        }
        return maxIndex;
    }


    /**
     * This function creates an ArrayList of strings, creates a random number generator, adds 5 strings
     * to the ArrayList, and returns a random string from the ArrayList
     *
     * @return A random string from the arraylist
     */




}
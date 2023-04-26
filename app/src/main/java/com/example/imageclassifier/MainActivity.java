package com.example.imageclassifier;

// This is importing the necessary libraries for the activity.

import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.activity.result.ActivityResultCallback;
import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AppCompatActivity;

import org.pytorch.IValue;
import org.pytorch.LiteModuleLoader;
import org.pytorch.Module;
import org.pytorch.Tensor;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.FloatBuffer;
import java.nio.file.Files;
import java.util.Random;
//import org.pytorch.IValue;
//import org.pytorch.torchvision.TensorImageUtils;

public class MainActivity extends AppCompatActivity {
    // Declaring the variables for the buttons and textview.
    Button loadBtn;
    Button classifybtn;
    ImageView img;
    TextView rndm;
    Bitmap bm;

    Module module;

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
        classifybtn = findViewById(R.id.classify_btn);
        img = findViewById(R.id.loadImage);
        rndm = findViewById(R.id.loadText);


        // This is registering the launcher for the activity result.
        launcher = registerForActivityResult(new ActivityResultContracts.GetContent(), new ActivityResultCallback<Uri>() {
            // This is the onActivityResult function. It is called when the launcher is launched. It
            // takes the uri of the image and sets the imageview to the image.
            @Override
            public void onActivityResult(Uri uri) {
                try {

                    bm = MediaStore.Images.Media.getBitmap(getContentResolver(), uri);
                    img.setImageBitmap(bm);
                } catch (IOException e) {
                    throw new RuntimeException(e);
                }
            }
        });

        // This is setting the onClickListener for the loadBtn button. When the button is clicked, the
        // launcher is launched and the randomStringGen function is called.
        loadBtn.setOnClickListener(view -> launcher.launch("image/*"));


        try {
            module = LiteModuleLoader.load(assetFilePath("model.pt"));
        } catch (IOException e) {
            throw new RuntimeException(e);
        }


        classifybtn.setOnClickListener(view -> {
            //  IValue output.forward(IValue.from(InputTensor));
            // Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(bm, TensorImageUtils.TORCHVISION_NORM_MEAN_RGB,TensorImageUtils.TORCHVISION_NORM_STD_RGB);
            //converting img to a bitmap then to a tensor
            //Bitmap bitmap = Bitmap.createBitmap(img.getWidth(), img.getHeight(), Bitmap.Config.ARGB_8888);
            FloatBuffer floatBuffer = FloatBuffer.allocate(bm.getByteCount());
            bm.copyPixelsToBuffer(floatBuffer);
            float[] floatArray = floatBuffer.array();

           // long[] shape = {1, bm.getHeight(), bm.getWidth(), 3}; // assuming RGB image
            long[] inputShape = {1, 3, 28, 28}; // assuming input shape is (batch_size, num_channels, height, width)
            long[] outputShape = {1, 10}; // assuming output shape is (batch_size, num_classes)

            // Convert the input image to a PyTorch tensor
            Tensor inputTensor = Tensor.fromBlob(floatBuffer, inputShape);

            // Call the model on the input tensor and obtain the output tensor
            Tensor outputTensor = module.forward(IValue.from(inputTensor)).toTensor();

            // Convert the output tensor to a Java float array
            float[] outputArray = outputTensor.getDataAsFloatArray();

            int predictedIndex = getMaxIndex(outputArray);

            String predictedClassName = ClassLabels.LABELS[getMaxIndex(outputArray)];

            rndm.setText(predictedIndex);
        });


        //String fileContents = loadAssetFile("myFile.txt");

        //AssetManager assetManager = getAssets();
        //FloatBuffer floatBuffer = ByteBuffer.wrap(byteArray).order(ByteOrder.nativeOrder()).asFloatBuffer();
        //Tensor tensor = Tensor.fromBlob(floatBuffer, shape)

    }
    public String assetFilePath (String assetName) throws IOException {
        File file = new File(getFilesDir(), assetName);
        if (file.exists()) {
            return file.getAbsolutePath();
        }

        try (InputStream inputStream = getAssets().open(assetName); OutputStream outputStream = Files.newOutputStream(file.toPath())) {
            byte[] buffer = new byte[1024];
            int read;
            while ((read = inputStream.read(buffer)) != -1) {
                outputStream.write(buffer, 0, read);
            }
            return file.getAbsolutePath();
        }
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
}

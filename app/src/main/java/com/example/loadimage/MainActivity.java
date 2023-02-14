package com.example.loadimage;

// This is importing the necessary libraries for the activity.
import androidx.activity.result.ActivityResultCallback;
import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContract;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AppCompatActivity;

import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Random;

public class MainActivity extends AppCompatActivity {
    // Declaring the variables for the buttons and textview.
    Button loadBtn;
    ImageView img;
    TextView rndm;
    Bitmap bm;


    ActivityResultLauncher<String> launcher;



    // This is the onCreate function. It is called when the activity is created. It is setting the
    // content view to the activity_main.xml file.
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // This is getting the buttons and textview from the xml file.
        loadBtn = findViewById(R.id.load_btn);
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
                rndm.setText(randomStringGen());
            }
        });




    }

    /**
     * This function creates an ArrayList of strings, creates a random number generator, adds 5 strings
     * to the ArrayList, and returns a random string from the ArrayList
     *
     * @return A random string from the arraylist
     */
    public String randomStringGen (){
        ArrayList<String> r_strings= new ArrayList<>();
        Random ran = new Random();

        r_strings.add("Ketchup");
        r_strings.add("Mustard");
        r_strings.add("Mayonnaise");
        r_strings.add("Salsa");
        r_strings.add("BBQ");
        r_strings.add("Cheese");


        return r_strings.get(ran.nextInt(r_strings.size()));
    }

}
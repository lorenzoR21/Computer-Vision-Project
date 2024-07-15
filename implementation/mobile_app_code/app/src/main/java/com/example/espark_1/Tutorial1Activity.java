package com.example.espark_1;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.AppCompatButton;

import android.content.Intent;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.TextView;
import android.widget.Toast;

import com.google.android.gms.tasks.OnCompleteListener;
import com.google.android.gms.tasks.Task;
import com.google.firebase.messaging.FirebaseMessaging;

import org.json.JSONObject;

import java.util.Map;

public class Tutorial1Activity extends AppCompatActivity {
    AppCompatButton next;
    TextView a;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_tutorial1);
        a = findViewById(R.id.btnRegister);
        next = findViewById(R.id.btnnext);
        next.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v)
            {
                Intent intent = new Intent(Tutorial1Activity.this, Tutorial2Activity.class);
                //Intent intent = new Intent(Tutorial1Activity.this, ProvaActivity.class);
                startActivity(intent);
            }
        });



       /* Intent intent = getIntent();
        if (intent != null && intent.getAction() != null && intent.getAction().equals("OPEN_ACTIVITY_2")) {
            Toast.makeText(getApplicationContext(),
                            "IF",
                            Toast.LENGTH_LONG)
                    .show();
            // If opened from notification, redirect to specific activity
            Intent specificIntent = new Intent(this, LoginActivity.class);
            startActivity(specificIntent);

            //finish(); // Finish the launcher activity to prevent going back to it
        } else {
            Toast.makeText(getApplicationContext(),
                            "ELSE",
                            Toast.LENGTH_LONG)
                    .show();
            // If not opened from notification, continue with normal flow
            // For example, redirect to login activity or main activity
            // Intent loginIntent = new Intent(this, LoginActivity.class);
            // startActivity(loginIntent);
            // Finish the launcher activity to prevent going back to it
            //finish();
        }*/

       /* FirebaseMessaging.getInstance().getToken()
                .addOnCompleteListener(new OnCompleteListener<String>() {
                    @Override
                    public void onComplete(@NonNull Task<String> task) {
                        if (!task.isSuccessful()) {
                            //Log.w(TAG, "Fetching FCM registration token failed", task.getException());
                            return;
                        }

                        // Get new FCM registration token
                        String token = task.getResult();

                        // Log and toast
                        //String msg = getString(R.string.msg_token_fmt, token);
                        a.setText(token);
                        Log.d("token", token);
                        //Toast.makeText(Tutorial1Activity.this, token, Toast.LENGTH_SHORT).show();
                    }
                });*/
    }
    //disable back button
    @Override
    public void onBackPressed(){}
}
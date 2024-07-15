package com.example.espark_1;

import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.AppCompatButton;

import android.content.Intent;
import android.os.Bundle;
import android.view.View;

public class Tutorial3Activity extends AppCompatActivity {
    AppCompatButton next;
    AppCompatButton skip;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_tutorial3);
        next = findViewById(R.id.btnnext);
        next.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v)
            {
                Intent intent = new Intent(Tutorial3Activity.this, Tutorial4Activity.class);
                startActivity(intent);
            }
        });

        skip = findViewById(R.id.btnskip);
        skip.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v)
            {
                Intent intent = new Intent(Tutorial3Activity.this, LoginActivity.class);
                startActivity(intent);
            }
        });
    }
    //disable back button
    @Override
    public void onBackPressed(){}
}
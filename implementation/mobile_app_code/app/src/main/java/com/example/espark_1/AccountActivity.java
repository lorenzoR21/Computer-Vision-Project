package com.example.espark_1;

import android.content.Intent;
import android.os.Bundle;
import android.view.View;

import androidx.appcompat.widget.AppCompatButton;

import com.example.espark_1.databinding.ActivityAccountBinding;

public class AccountActivity extends DrawerBaseActivity {

    private ActivityAccountBinding binding;
    private AppCompatButton btn_profile;
    private AppCompatButton btn_pay;
    private AppCompatButton btn_vehicle;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        binding = ActivityAccountBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());


        btn_profile = findViewById(R.id.btn_profile);
        btn_pay = findViewById(R.id.btn_pay);
        btn_vehicle = findViewById(R.id.btn_vehicle);

        btn_profile.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(AccountActivity.this, ProfileActivity.class);
                startActivity(intent);
            }
        });
        btn_vehicle.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(AccountActivity.this, VehicleActivity.class);
                startActivity(intent);
            }
        });
        btn_pay.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(AccountActivity.this, CardActivity.class);
                startActivity(intent);
            }
        });
    }
    //disable back button
    @Override
    public void onBackPressed(){}
}
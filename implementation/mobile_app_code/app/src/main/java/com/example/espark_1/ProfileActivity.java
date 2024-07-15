package com.example.espark_1;

import android.content.Intent;
import android.os.Bundle;
import android.util.Log;
import android.view.View;

import androidx.appcompat.widget.AppCompatEditText;
import androidx.appcompat.widget.AppCompatImageView;
import androidx.appcompat.widget.AppCompatTextView;

import com.example.espark_1.databinding.ActivityProfileBinding;
import com.google.firebase.auth.FirebaseAuth;

public class ProfileActivity extends DrawerBaseActivity {

    private ActivityProfileBinding binding;
    private AppCompatEditText name;
    private AppCompatEditText email;
    private AppCompatEditText phone;
    private AppCompatEditText password;
    FirebaseAuth mAuth;
    /*AppCompatImageView btn_name;
    AppCompatImageView btn_email;
    AppCompatImageView btn_password;*/
    //AppCompatImageView btn_phone;
    AppCompatImageView btn_exit;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        binding = ActivityProfileBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());

        mAuth = FirebaseAuth.getInstance();

        name = findViewById(R.id.txt_profile);
        email = findViewById(R.id.txt_email);
        //phone = findViewById(R.id.txt_number);
        password = findViewById(R.id.txt_password);

        /*btn_name = findViewById(R.id.btn_edit1);
        btn_email = findViewById(R.id.btn_edit2);
        btn_password = findViewById(R.id.btn_edit3);*/
        //btn_phone = findViewById(R.id.btn_edit4);
        btn_exit = findViewById(R.id.image_exit);
        btn_exit.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(ProfileActivity.this, AccountActivity.class);
                startActivity(intent);
            }
        });

        name.setText(User.name() + " " + User.surname());
        email.setText(User.email());
        //phone.setText(User.telNumber());
        password.setText("********");

        name.setEnabled(false);
        email.setEnabled(false);
        //phone.setEnabled(false);
        password.setEnabled(false);
    }
    //disable back button
    @Override
    public void onBackPressed(){}
}
package com.example.espark_1;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.AppCompatButton;
import androidx.appcompat.widget.AppCompatImageView;

import android.content.Intent;
import android.os.Bundle;
import android.text.method.HideReturnsTransformationMethod;
import android.text.method.PasswordTransformationMethod;
import android.util.Log;
import android.view.View;
import android.widget.CheckBox;
import android.widget.EditText;
import android.widget.TextView;
import android.widget.Toast;

import com.google.android.gms.tasks.OnCompleteListener;
import com.google.android.gms.tasks.Task;
import com.google.firebase.auth.AuthResult;
import com.google.firebase.auth.FirebaseAuth;
import com.google.firebase.auth.FirebaseUser;
import com.google.firebase.database.DatabaseReference;
import com.google.firebase.database.FirebaseDatabase;

public class SigninActivity extends AppCompatActivity {

    private FirebaseDatabase database;
    private DatabaseReference myRef;
    private FirebaseAuth mAuth;

    EditText txt_email;
    EditText txt_password;
    AppCompatButton btn_sign_up;
    TextView txt_sign_in;
    AppCompatImageView btn_cancel;
    private CheckBox showPasswordCheckBox;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_signin);

        mAuth = FirebaseAuth.getInstance();

        txt_email = findViewById(R.id.email);
        txt_password = findViewById(R.id.password);
        btn_sign_up = findViewById(R.id.btn_next);
        txt_sign_in = findViewById(R.id.txt_login);
        btn_cancel = findViewById(R.id.image_exit);

        btn_cancel.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v)
            {
                Intent intent = new Intent(SigninActivity.this, LoginActivity.class);
                startActivity(intent);
            }
        });

        txt_sign_in.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v)
            {
                Intent intent = new Intent(SigninActivity.this, LoginActivity.class);
                startActivity(intent);
            }
        });

        btn_sign_up.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v)
            {
                register();
            }
        });
        txt_password.setTransformationMethod(PasswordTransformationMethod.getInstance());
        showPasswordCheckBox = findViewById(R.id.showPasswordCheckbox);
        showPasswordCheckBox.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                // Check the checkbox state
                boolean isChecked = showPasswordCheckBox.isChecked();

                // Toggle password visibility
                if (isChecked) {
                    // Show password
                    txt_password.setTransformationMethod(HideReturnsTransformationMethod.getInstance());
                } else {
                    // Hide password
                    txt_password.setTransformationMethod(PasswordTransformationMethod.getInstance());
                }

                // Move the cursor to the end of the text
                txt_password.setSelection(txt_password.getText().length());
            }
        });
    }

    void register()
    {
        String email = txt_email.getText().toString();
        String password = txt_password.getText().toString();
        if (email.isEmpty() || password.isEmpty())
        {
            return;
        }

        mAuth.createUserWithEmailAndPassword(email, password)
                .addOnCompleteListener(this, new OnCompleteListener<AuthResult>() {
                    @Override
                    public void onComplete(@NonNull Task<AuthResult> task) {
                        if (task.isSuccessful()) {
                            // User registration successful
                            Toast.makeText(getApplicationContext(),
                                            "Register successful!",
                                            Toast.LENGTH_LONG)
                                    .show();

                            Intent intent = new Intent(SigninActivity.this, RegisterProfileActivity.class);
                            FirebaseUser user = mAuth.getCurrentUser();
                            String uid = user.getUid();
                            //Log.d("REGISTER PAGE 1", uid);
                            intent.putExtra("uid", uid);
                            intent.putExtra("email", email);
                            intent.putExtra("password", password);
                            intent.putExtra("user", user);
                            startActivity(intent);
                        } else {
                            // User registration failed

                            Toast.makeText(getApplicationContext(),
                                            "Register failed!",
                                            Toast.LENGTH_LONG)
                                    .show();
                            /*Intent intent = new Intent(SigninActivity.this, LoginActivity.class);
                            startActivity(intent);*/
                        }
                    }
                });
    }
    //disable back button
    @Override
    public void onBackPressed(){}
}


package com.example.espark_1;

import static android.content.ContentValues.TAG;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.AppCompatButton;

import com.example.espark_1.User;

import android.annotation.SuppressLint;
import android.content.Intent;
import android.os.Bundle;
import android.text.TextUtils;
import android.text.method.HideReturnsTransformationMethod;
import android.text.method.PasswordTransformationMethod;
import android.util.Log;
import android.util.Pair;
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
import com.google.firebase.database.DataSnapshot;
import com.google.firebase.database.DatabaseError;
import com.google.firebase.database.DatabaseReference;
import com.google.firebase.database.FirebaseDatabase;
import com.google.firebase.database.ValueEventListener;

import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;

public class LoginActivity extends AppCompatActivity
{
    private FirebaseDatabase database;
    private DatabaseReference myRef;
    private EditText emailTextView, passwordTextView;
    private TextView btn_register;
    private TextView btn_forgot_pass;
    private AppCompatButton btn_login;
    //private ProgressBar progressbar;
    private CheckBox showPasswordCheckBox;
    private FirebaseAuth mAuth;

    @SuppressLint("MissingInflatedId")
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_login);
        mAuth = FirebaseAuth.getInstance();

        // initialising all views through id defined above
        emailTextView = findViewById(R.id.email);
        passwordTextView = findViewById(R.id.password);
        btn_login = findViewById(R.id.btnLogin);
        btn_register = findViewById(R.id.btnRegister);
        btn_forgot_pass  = findViewById(R.id.btnForgot);

        btn_forgot_pass.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v)
            {
                Intent intent = new Intent(LoginActivity.this, ForgotPasswordActivity.class);
                startActivity(intent);
            }
        });

        // Set on Click Listener on Sign-in button
        btn_register.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v)
            {
                Intent intent = new Intent(LoginActivity.this, SigninActivity.class);
                startActivity(intent);
            }
        });


        btn_login.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                /*Intent intent = new Intent(LoginActivity.this, HomepageActivity.class);
                startActivity(intent);*/
                loginUserAccount();
            }
        });
        passwordTextView.setTransformationMethod(PasswordTransformationMethod.getInstance());
        showPasswordCheckBox = findViewById(R.id.showPasswordCheckbox);
        showPasswordCheckBox.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                // Check the checkbox state
                boolean isChecked = showPasswordCheckBox.isChecked();

                // Toggle password visibility
                if (isChecked) {
                    // Show password
                    passwordTextView.setTransformationMethod(HideReturnsTransformationMethod.getInstance());
                } else {
                    // Hide password
                    passwordTextView.setTransformationMethod(PasswordTransformationMethod.getInstance());
                }

                // Move the cursor to the end of the text
                passwordTextView.setSelection(passwordTextView.getText().length());
            }
        });

    }

    public interface DataExtractionCallback {
        void onDataExtracted();
    }

    private void loginUserAccount()
    {

        // show the visibility of progress bar to show loading
        //progressbar.setVisibility(View.VISIBLE);

        // Take the value of two edit texts in Strings
        String email, password;
        email = emailTextView.getText().toString();
        password = passwordTextView.getText().toString();

        // validations for input email and password
        if (TextUtils.isEmpty(email)) {
            Toast.makeText(getApplicationContext(),
                            "Please enter email!!",
                            Toast.LENGTH_LONG)
                    .show();
            return;
        }

        if (TextUtils.isEmpty(password)) {
            Toast.makeText(getApplicationContext(),
                            "Please enter password!!",
                            Toast.LENGTH_LONG)
                    .show();
            return;
        }

        // signin existing user
        mAuth.signInWithEmailAndPassword(email, password)
                .addOnCompleteListener(
                        new OnCompleteListener<AuthResult>() {
                            @Override
                            public void onComplete(
                                    @NonNull Task<AuthResult> task)
                            {
                                if (task.isSuccessful()) {
                                    Toast.makeText(getApplicationContext(),
                                                    "Login successful!",
                                                    Toast.LENGTH_LONG)
                                            .show();
                                    FirebaseUser user = mAuth.getCurrentUser();
                                    if (user != null) {
                                        String uid = user.getUid();
                                        allocateUser(uid);
                                    }
                                }

                                else {

                                    // sign-in failed
                                    Toast.makeText(getApplicationContext(),
                                                    "Login failed!!",
                                                    Toast.LENGTH_LONG)
                                            .show();

                                    // hide the progress bar
                                    //progressbar.setVisibility(View.GONE);
                                }
                            }
                        });
    }

    private void allocateUser(String uid/*, final DataExtractionCallback callback*/)
    {
        database = FirebaseDatabase.getInstance();
        myRef = database.getReference("user");

        myRef.addListenerForSingleValueEvent(new ValueEventListener() {
            @Override
            public void onDataChange(DataSnapshot dataSnapshot) {
                for (DataSnapshot childSnapshot : dataSnapshot.getChildren())
                {
                    String id = childSnapshot.getKey();
                    String userUID = childSnapshot.child("uid").getValue(String.class);
                    if (userUID.equals(uid)) {
                        User.set_id(id);
                        User.set_uid(userUID);
                        Log.d("LOGIN PAGE 2", String.valueOf(User.id()));
                        User.set_name(childSnapshot.child("name").getValue(String.class));
                        //Log.d("LOGIN PAGE 2", String.valueOf(User.name()));
                        User.set_surname(childSnapshot.child("surname").getValue(String.class));
                        User.set_email(childSnapshot.child("email").getValue(String.class));
                        //User.set_password(childSnapshot.child("password").getValue(String.class));
                        User.set_age(childSnapshot.child("age").getValue(Integer.class));
                        //User.set_telNumber(childSnapshot.child("telNumber").getValue(String.class));

                        List<Pair<String, String>> vehicleList = new ArrayList<>();
                        for (DataSnapshot lineSnapshot : childSnapshot.child("vehicle").getChildren())
                        {
                            String a = lineSnapshot.getKey();
                            String l = lineSnapshot.child("license").getValue(String.class);
                            String m = lineSnapshot.child("model").getValue(String.class);
                            Pair<String, String> v = new Pair<>(l, m);
                            //Log.d("LOGIN PAGE 2", String.valueOf(l));
                            vehicleList.add(v);
                        }
                        User.set_vehicle(vehicleList);

                        List<Pair<String, String>> payList = new ArrayList<>();
                        for (DataSnapshot lineSnapshot : childSnapshot.child("methodPay").getChildren())
                        {
                            String a = lineSnapshot.getKey();
                            Pair<String, String> p = new Pair<>(lineSnapshot.child("numCard").getValue(String.class), lineSnapshot.child("type").getValue(String.class));
                            payList.add(p);
                        }
                        User.set_payMethod(payList);

                        List<Pair<Pair<String, String>, String>> slotList = new ArrayList<>();
                        for (DataSnapshot lineSnapshot : childSnapshot.child("selectedSlot").getChildren())
                        {
                            String a = lineSnapshot.getKey();
                            Pair<String, String> s = new Pair<>(lineSnapshot.child("parkingName").getValue(String.class), lineSnapshot.child("slotNum").getValue(String.class));
                            Pair<Pair<String, String>, String> s1 = new Pair<>(s,lineSnapshot.child("parkingId").getValue(String.class));
                            slotList.add(s1);
                        }
                        User.set_selectedSlot(slotList);

                        List<Pair<String, String> > favouriteList = new ArrayList<>();
                        for (DataSnapshot lineSnapshot : childSnapshot.child("favouriteParking").getChildren())
                        {
                            String a = lineSnapshot.getKey();
                            Pair<String, String> s = new Pair<>(a, lineSnapshot.child("name").getValue(String.class));
                            favouriteList.add(s);
                        }
                        User.set_favouriteParking(favouriteList);

                        List<ParkingStop> ongoingList = new ArrayList<>();
                        for (DataSnapshot lineSnapshot : childSnapshot.child("ongoingParking").getChildren())
                        {
                            String a = lineSnapshot.getKey();
                            ParkingStop ps = new ParkingStop();
                            ps.set_id(a);
                            ps.set_name(lineSnapshot.child("parkingName").getValue(String.class));
                            ps.set_prize(lineSnapshot.child("prize").getValue(Float.class));
                            ps.set_vehicleLP(lineSnapshot.child("vehicleLP").getValue(String.class));
                            ps.set_payCard(lineSnapshot.child("payCard").getValue(String.class));
                            ps.set_costHourly(lineSnapshot.child("costH").getValue(Float.class));
                            ps.set_ongoing(lineSnapshot.child("ongoing").getValue(String.class));
                            SimpleDateFormat format = new SimpleDateFormat("dd-MM-yyyy HH:mm");
                            String d = lineSnapshot.child("startDate").getValue(String.class);
                            //Log.d("LOGIN PAGE 2", String.valueOf(d));
                            try {
                                Date date = format.parse(d);
                                //Log.d("Extend Page", String.valueOf(date.getDate()));
                                //System.out.println("Parsed Date-Time: " + date);
                                ps.set_startDate(date);
                            } catch (ParseException e) {
                                throw new RuntimeException(e);
                            }
                            String d1 = lineSnapshot.child("endDate").getValue(String.class);
                            //Log.d("LOGIN PAGE 2", String.valueOf(d1));
                            try {
                                Date date = format.parse(d1);
                                //Log.d("LOGIN PAGE 2", String.valueOf(String.valueOf(date.getMinutes()).length()));
                                //System.out.println("Parsed Date-Time: " + date);
                                ps.set_endDate(date);
                            } catch (ParseException e) {
                                throw new RuntimeException(e);
                            }
                            ongoingList.add(ps);

                        }
                        User.set_ongoing(ongoingList);

                        List<ParkingStop> historyList = new ArrayList<>();
                        for (DataSnapshot lineSnapshot : childSnapshot.child("parkingHistory").getChildren())
                        {
                            String a = lineSnapshot.getKey();
                            ParkingStop ps = new ParkingStop();
                            ps.set_id(a);
                            ps.set_name(lineSnapshot.child("parkingName").getValue(String.class));
                            ps.set_prize(lineSnapshot.child("prize").getValue(Float.class));
                            SimpleDateFormat format=new SimpleDateFormat("dd-MM-yyyy HH:mm");
                            String d = lineSnapshot.child("startDate").getValue(String.class);
                            try {
                                Date date = format.parse(d);

                                ps.set_startDate(date);
                            } catch (ParseException e) {
                                throw new RuntimeException(e);
                            }
                            String d1 = lineSnapshot.child("endDate").getValue(String.class);
                            try {
                                Date date = format.parse(d1);
                                ps.set_endDate(date);
                            } catch (ParseException e) {
                                throw new RuntimeException(e);
                            }
                            historyList.add(ps);

                        }
                        User.set_history(historyList);

                        User.tokenUpdateDB();

                        break;
                    }
                }

                Intent intent = new Intent(LoginActivity.this, HomepageActivity.class);
                startActivity(intent);
            }

            @Override
            public void onCancelled(DatabaseError databaseError) {
                Log.e(TAG, "onCancelled", databaseError.toException());
            }
        });

        // Once data extraction is complete, trigger the callback
        //callback.onDataExtracted();
    }
    //disable back button
    @Override
    public void onBackPressed(){}
}
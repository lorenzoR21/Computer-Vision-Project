package com.example.espark_1;

import static android.content.ContentValues.TAG;

import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.AppCompatButton;
import androidx.appcompat.widget.AppCompatImageView;

import android.content.Intent;
import android.os.Bundle;
import android.util.Log;
import android.util.Pair;
import android.view.View;
import android.widget.EditText;

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

public class RegisterProfileActivity extends AppCompatActivity {

    private FirebaseDatabase database;
    private DatabaseReference myRef;
    private FirebaseUser user;
    String user_uid;
    String user_email;
    String user_password;
    AppCompatImageView btn_back;
    EditText txt_name;
    EditText txt_surname;
    EditText txt_date;
    //EditText txt_tel;
    AppCompatButton btn_next;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_register_profile);

        Bundle data = getIntent().getExtras();
        user_uid = (String) data.get("uid");
        user = (FirebaseUser) data.get("user");
        user_email = (String) data.get("email");
        user_password = (String) data.get("password");

        txt_name = findViewById(R.id.ed_name);
        txt_surname = findViewById(R.id.ed_surname);
        txt_date = findViewById(R.id.ed_date);
        //txt_tel = findViewById(R.id.ed_tel);
        btn_next = findViewById(R.id.btn_next);
        btn_back = findViewById(R.id.image_exit);

        btn_back.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v)
            {
                user.delete();
                Intent intent = new Intent(RegisterProfileActivity.this, SigninActivity.class);
                startActivity(intent);
            }
        });

        btn_next.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v)
            {
                create_new_user();
            }
        });
    }

    void create_new_user()
    {
        String name = txt_name.getText().toString();
        String surname = txt_surname.getText().toString();
        Integer date = Integer.valueOf(txt_date.getText().toString());
        //String tel = txt_tel.getText().toString();

        database = FirebaseDatabase.getInstance();
        myRef = database.getReference("user");

        myRef.addListenerForSingleValueEvent(new ValueEventListener() {
            @Override
            public void onDataChange(DataSnapshot dataSnapshot) {
                int i = 1;
                for (DataSnapshot childSnapshot : dataSnapshot.getChildren()) {
                    String id = childSnapshot.getKey();
                    i++;
                }

                myRef.child("u" + String.valueOf(i)).child("uid").setValue(user_uid);
                myRef.child("u" + String.valueOf(i)).child("name").setValue(name);
                myRef.child("u" + String.valueOf(i)).child("surname").setValue(surname);
                //myRef.child("u" + String.valueOf(i)).child("telNumber").setValue(tel);
                myRef.child("u" + String.valueOf(i)).child("age").setValue(date);
                myRef.child("u" + String.valueOf(i)).child("email").setValue(user_email);
                //myRef.child("u" + String.valueOf(i)).child("password").setValue(user_password);

                User.set_id("u" + String.valueOf(i));
                User.set_uid(user_uid);
                User.set_name(name);
                User.set_surname(surname);
                User.set_age(date);
                //User.set_telNumber(tel);
                User.set_email(user_email);
               // User.set_password(user_password);
                List<Pair<String, String>> vehicleList = new ArrayList<>();
                User.set_vehicle(vehicleList);

                List<Pair<String, String>> payList = new ArrayList<>();
                User.set_payMethod(payList);

                List<Pair<Pair<String, String>, String>> slotList = new ArrayList<>();
                User.set_selectedSlot(slotList);

                List<Pair<String, String> > favouriteList = new ArrayList<>();
                User.set_favouriteParking(favouriteList);

                List<ParkingStop> ongoingList = new ArrayList<>();
                User.set_ongoing(ongoingList);

                List<ParkingStop> historyList = new ArrayList<>();
                User.set_history(historyList);

                User.tokenUpdateDB();


                Intent intent = new Intent(RegisterProfileActivity.this, HomepageActivity.class);
                startActivity(intent);
            }

            @Override
            public void onCancelled(DatabaseError databaseError) {
                Log.e(TAG, "onCancelled", databaseError.toException());
            }
        });

    }
    //disable back button
    @Override
    public void onBackPressed(){}
}
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
import android.widget.TextView;

import com.google.firebase.database.DataSnapshot;
import com.google.firebase.database.DatabaseError;
import com.google.firebase.database.DatabaseReference;
import com.google.firebase.database.FirebaseDatabase;
import com.google.firebase.database.ValueEventListener;

import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.Date;
import java.util.List;

public class SummaryParking extends AppCompatActivity {

    TextView txt_time;
    TextView txt_start;
    TextView txt_expiry;
    TextView txt_vehicle;
    TextView txt_name;
    TextView txt_pay;
    TextView txt_tot;
    private AppCompatButton btn_next;
    private AppCompatImageView btn_pre;

    Integer tot_time;
    int expiry_hour;
    int expiry_minute;
    float tot_prize;
    String name;
    int vehicleId;
    int cardId;
    Calendar calendar;
    int current_day;
    int current_month;
    int current_year;
    int current_hour;
    int current_minute;

    int calendarStart_Day;
    int calendarStart_Month;
    int calendarEnd_Day;
    int calendarEnd_Month;
    private FirebaseDatabase database;
    private DatabaseReference myRef;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_summary_parking);



        Bundle data = getIntent().getExtras();
        tot_time = (Integer) data.getInt("time");
        expiry_hour = data.getInt("hourExpiry");
        expiry_minute = data.getInt("minuteExpiry");
        tot_prize = data.getFloat("tot");
        name = data.getString("name");
        cardId = data.getInt("card");
        vehicleId = data.getInt("vehicle");
        current_hour = data.getInt("currentHour");
        current_minute = data.getInt("currentMinute");
        calendarStart_Day = data.getInt("calendarStart_Day");
        calendarStart_Month = data.getInt("calendarStart_Month");
        calendarEnd_Day = data.getInt("calendarEnd_Day");
        calendarEnd_Month = data.getInt("calendarEnd_Month");

        btn_pre = findViewById(R.id.btn_cancel);
        btn_pre.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {

                Intent intent = new Intent(SummaryParking.this, HomepageActivity.class);
                startActivity(intent);
            }
        });

        txt_time = findViewById(R.id.txt_time);
        if (tot_time % 2 == 0)
        {
            txt_time.setText(String.valueOf(tot_time / 2) + " hr");
        }
        else
        {
            int a = tot_time / 2;
            txt_time.setText(String.valueOf(a) + ":30" + " hr");
        }

        calendar = Calendar.getInstance();
        /*current_day = calendar.get(Calendar.DAY_OF_MONTH);
        current_month = calendar.get(Calendar.MONTH);*/
        current_year = calendar.get(Calendar.YEAR);

        txt_start = findViewById(R.id.txt_start);
        txt_start.setText(String.valueOf(calendarStart_Day) + "-" + String.valueOf(calendarStart_Month) + " " + String.valueOf(current_hour) + ":" + String.valueOf(current_minute));

        txt_expiry = findViewById(R.id.txt_expiry);
        txt_expiry.setText(String.valueOf(calendarEnd_Day) + "-" + String.valueOf(calendarEnd_Month) + " " + String.valueOf(expiry_hour) + ":" + String.valueOf(expiry_minute));


        txt_vehicle = findViewById(R.id.txt_vehicle);
        txt_vehicle.setText(User.vehicle().get(vehicleId).second + " (" + User.vehicle().get(vehicleId).first + ")");

        txt_name = findViewById(R.id.txt_name);
        txt_name.setText(name);

        txt_pay = findViewById(R.id.txt_pay);
        txt_pay.setText(User.payMethod().get(cardId).first + ", " + User.payMethod().get(cardId).second);

        txt_tot = findViewById(R.id.txt_tot);
        txt_tot.setText("€" + String.valueOf(tot_prize));

        btn_next = findViewById(R.id.btn_next);
        btn_next.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                //add info on database
                updateOngoingParking();

                Intent intent = new Intent(SummaryParking.this, HomepageActivity.class);
                startActivity(intent);
            }
        });

    }

    void updateOngoingParking()
    {
        database = FirebaseDatabase.getInstance();
        myRef = database.getReference("user").child(User.id()).child("ongoingParking");
        myRef.addListenerForSingleValueEvent(new ValueEventListener() {
            @Override
            public void onDataChange(DataSnapshot dataSnapshot) {
                int i = 0;
                for (DataSnapshot childSnapshot : dataSnapshot.getChildren())
                {
                    String id = childSnapshot.getKey();
                    i++;
                }

                float p = 2F; // modificare
                myRef.child("p1").child("parkingName").setValue(name);
                myRef.child("p1").child("prize").setValue(tot_prize);
                myRef.child("p1").child("vehicleLP").setValue(User.vehicle().get(vehicleId).first);
                myRef.child("p1").child("payCard").setValue(User.payMethod().get(cardId).first);
                myRef.child("p1").child("ongoing").setValue("true");
                myRef.child("p1").child("costH").setValue(p);

                String currentDate = String.valueOf(calendarStart_Day) + "-" + String.valueOf(calendarStart_Month) + "-" + String.valueOf(current_year) + " " + String.valueOf(current_hour) + ":" + String.valueOf(current_minute);
                myRef.child("p1").child("startDate").setValue(currentDate);

                String expiryDate = String.valueOf(calendarEnd_Day) + "-" + String.valueOf(calendarEnd_Month) + "-" + String.valueOf(current_year) + " " + String.valueOf(expiry_hour) + ":" + String.valueOf(expiry_minute);
                myRef.child("p1").child("endDate").setValue(expiryDate);

                User.userUpdate();
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
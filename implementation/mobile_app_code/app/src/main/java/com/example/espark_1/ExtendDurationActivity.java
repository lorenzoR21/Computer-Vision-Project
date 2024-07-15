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
import android.widget.RadioButton;
import android.widget.RadioGroup;
import android.widget.TextView;

import com.google.firebase.database.DataSnapshot;
import com.google.firebase.database.DatabaseError;
import com.google.firebase.database.DatabaseReference;
import com.google.firebase.database.FirebaseDatabase;
import com.google.firebase.database.ValueEventListener;

import java.text.SimpleDateFormat;
import java.util.Calendar;
import java.util.Date;

public class ExtendDurationActivity extends AppCompatActivity {

    AppCompatButton btn_minus;
    AppCompatButton btn_plus;
    AppCompatImageView btn_cancel;
    AppCompatButton btn_next;
    TextView txt_time;
    TextView txt_startH;
    TextView txt_endH;
    TextView txt_newEndH;
    TextView txt_total;
    Integer tot_stop_time;
    Calendar calendar;
    int current_hour;
    int current_minute;
    int final_hour;
    int final_minute;
    float total_prize;
    Date current_date;
    Date final_date;
    boolean add_card = false;
    private RadioGroup radioGroup;
    AppCompatImageView btn_add;
    EditText txt_new_type;
    EditText txt_new_num;
    int cardId;
    private FirebaseDatabase database;
    private DatabaseReference myRef;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_extend_duration);

        cardId = -1;

        total_prize = 0;
        tot_stop_time = 0;
        current_date = User.ongoing().get(0).endDate();
        //Log.d("Extend Page", String.valueOf(current_date.getDay()));

        btn_add = findViewById(R.id.btn_add);
        txt_new_type= findViewById(R.id.ed_type);
        txt_new_num= findViewById(R.id.ed_num);
        txt_new_type.setVisibility(View.GONE);
        txt_new_num.setVisibility(View.GONE);

        btn_cancel = findViewById(R.id.btn_cancel);
        btn_cancel.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                tot_stop_time = 0;
                total_prize = 0;
                Intent intent = new Intent(ExtendDurationActivity.this, Favorite_parkingActivity.class);
                intent.putExtra("section", "ongoing");
                startActivity(intent);
            }
        });

        //next iff user select time and add vehicle details
        btn_next = findViewById(R.id.btn_next);
        btn_next.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                updateOngoingParking();
            }
        });
        btn_next.setEnabled(false);

        txt_time = findViewById(R.id.txt_time);
        txt_time.setText("0 hr");

        btn_minus = findViewById(R.id.btn_min);
        btn_minus.setEnabled(false);
        btn_minus.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (tot_stop_time > 0)
                {
                    tot_stop_time--;
                }
                if (tot_stop_time == 0)
                {
                    btn_minus.setEnabled(false);
                    btn_next.setEnabled(false);
                }

                if (tot_stop_time % 2 == 0)
                {
                    txt_time.setText(String.valueOf(tot_stop_time / 2) + " hr");
                }
                else
                {
                    int a = tot_stop_time / 2;
                    txt_time.setText(String.valueOf(a) + ":30" + " hr");
                }
                calendar.add(Calendar.MINUTE, -30);
                final_date = calendar.getTime();
                txt_newEndH.setText(String.valueOf(final_date.getHours()) + ":" + String.valueOf(final_date.getMinutes()));

                total_prize = total_prize - (User.ongoing().get(0).costHourly() / 2);
                txt_total.setText("€" + String.valueOf(total_prize));
            }
        });
        btn_plus = findViewById(R.id.btn_plus);
        btn_plus.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                btn_next.setEnabled(true);
                if (tot_stop_time == 0)
                {
                    btn_minus.setEnabled(true);
                }
                tot_stop_time++;

                if (tot_stop_time % 2 == 0)
                {
                    txt_time.setText(String.valueOf(tot_stop_time / 2) + " hr");
                }
                else
                {
                    int a = tot_stop_time / 2;
                    txt_time.setText(String.valueOf(a) + ":30" + " hr");
                }


                calendar.add(Calendar.MINUTE, 30);
                final_date = calendar.getTime();
                txt_newEndH.setText(String.valueOf(final_date.getHours()) + ":" + String.valueOf(final_date.getMinutes()));

                total_prize = total_prize + (User.ongoing().get(0).costHourly() / 2);
                txt_total.setText("€" + String.valueOf(total_prize));
            }
        });

        calendar = Calendar.getInstance();
        calendar.setTime(current_date);

        txt_startH = findViewById(R.id.txt_startH);
        if (String.valueOf(User.ongoing().get(0).startDate().getHours()).length() == 1)
        {
            if (String.valueOf(User.ongoing().get(0).startDate().getMinutes()).length() == 1)
            {
                txt_startH.setText("0" + String.valueOf(User.ongoing().get(0).startDate().getHours()) + ":0" + String.valueOf(User.ongoing().get(0).startDate().getMinutes()));
            }
            else
            {
                txt_startH.setText("0" + String.valueOf(User.ongoing().get(0).startDate().getHours()) + ":" + String.valueOf(User.ongoing().get(0).startDate().getMinutes()));
            }
        }
        else if (String.valueOf(User.ongoing().get(0).startDate().getMinutes()).length() == 1)
        {
            txt_startH.setText(String.valueOf(User.ongoing().get(0).startDate().getHours()) + ":0" + String.valueOf(User.ongoing().get(0).startDate().getMinutes()));
        }
        else
        {
            txt_startH.setText(String.valueOf(User.ongoing().get(0).startDate().getHours()) + ":" + String.valueOf(User.ongoing().get(0).startDate().getMinutes()));
        }

        txt_endH = findViewById(R.id.txt_endH);
        if (String.valueOf(User.ongoing().get(0).endDate().getHours()).length() == 1)
        {
            if (String.valueOf(User.ongoing().get(0).endDate().getMinutes()).length() == 1)
            {
                txt_endH.setText("0" + String.valueOf(User.ongoing().get(0).endDate().getHours()) + ":0" + String.valueOf(User.ongoing().get(0).endDate().getMinutes()));
            }
            else
            {
                txt_endH.setText("0" + String.valueOf(User.ongoing().get(0).endDate().getHours()) + ":" + String.valueOf(User.ongoing().get(0).endDate().getMinutes()));
            }
        }
        else if(String.valueOf(User.ongoing().get(0).endDate().getMinutes()).length() == 1)
        {
            txt_endH.setText(String.valueOf(User.ongoing().get(0).endDate().getHours()) + ":0" + String.valueOf(User.ongoing().get(0).endDate().getMinutes()));
        }
        else
        {
            txt_endH.setText(String.valueOf(User.ongoing().get(0).endDate().getHours()) + ":" + String.valueOf(User.ongoing().get(0).endDate().getMinutes()));
        }


        txt_newEndH = findViewById(R.id.txt_newEndH);
        if (String.valueOf(User.ongoing().get(0).endDate().getHours()).length() == 1)
        {
            if (String.valueOf(User.ongoing().get(0).endDate().getMinutes()).length() == 1)
            {
                txt_newEndH.setText("0" + String.valueOf(User.ongoing().get(0).endDate().getHours()) + ":0" + String.valueOf(User.ongoing().get(0).endDate().getMinutes()));
            }
            else
            {
                txt_newEndH.setText("0" + String.valueOf(User.ongoing().get(0).endDate().getHours()) + ":" + String.valueOf(User.ongoing().get(0).endDate().getMinutes()));
            }
        }
        else if (String.valueOf(User.ongoing().get(0).endDate().getMinutes()).length() == 1)
        {
            txt_newEndH.setText(String.valueOf(User.ongoing().get(0).endDate().getHours()) + ":0" + String.valueOf(User.ongoing().get(0).endDate().getMinutes()));
        }
        else
        {
            txt_newEndH.setText(String.valueOf(User.ongoing().get(0).endDate().getHours()) + ":" + String.valueOf(User.ongoing().get(0).endDate().getMinutes()));
        }

        txt_total = findViewById(R.id.txt_total);
        txt_total.setText("€0.00");

        if (cardId == -1)
        {
            btn_next.setEnabled(false);
        }

        radioGroup = findViewById(R.id.radioGroup);
        int iconResource = add_card ? R.drawable.ic_min : R.drawable.ic_add;
        btn_add.setImageResource(iconResource);

        for (int i = 0; i < User.payMethod().size(); i++) {
            RadioButton radioButton = new RadioButton(this);
            radioButton.setId(i); // Set a unique ID for each radio button
            radioButton.setText("Type: " + User.payMethod().get(i).second + ", Number: " + User.payMethod().get(i).first);
            radioGroup.addView(radioButton);
        }

        radioGroup.setOnCheckedChangeListener(new RadioGroup.OnCheckedChangeListener() {
            @Override
            public void onCheckedChanged(RadioGroup group, int checkedId) {

                RadioButton selectedRadioButton = findViewById(checkedId);
                if (selectedRadioButton != null) {
                    // Check if the selected RadioButton is already checked
                        /*if (selectedRadioButton.isChecked()) {
                            // Toggle the selection state
                            selectedRadioButton.setChecked(false);
                        }*/
                    String selectedOption = selectedRadioButton.getText().toString();
                    // Do something with the selected option
                    cardId = checkedId;
                    btn_next.setEnabled(true);
                    if (add_card) {
                        add_card = false;
                        btn_add.setImageResource(R.drawable.ic_add);
                        txt_new_type.setVisibility(View.GONE);
                        txt_new_num.setVisibility(View.GONE);
                    }
                }
            }
        });
        btn_add.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                add_card = !add_card;
                if (add_card) {
                    radioGroup.clearCheck();
                    btn_add.setImageResource(R.drawable.ic_min);
                    txt_new_type.setVisibility(View.VISIBLE);
                    txt_new_num.setVisibility(View.VISIBLE);
                    btn_next.setEnabled(true);
                }
                else {
                    btn_add.setImageResource(R.drawable.ic_add);
                    txt_new_type.setVisibility(View.GONE);
                    txt_new_num.setVisibility(View.GONE);
                    btn_next.setEnabled(false);
                }
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

                myRef.child("p1").child("prize").setValue(total_prize + User.ongoing().get(0).prize());

                String expiryDate = String.valueOf(final_date.getDate()) + "-" + String.valueOf(calendar.get(Calendar.MONTH) + 1) + "-" + String.valueOf(calendar.get(Calendar.YEAR)) + " " + String.valueOf(final_date.getHours()) + ":" + String.valueOf(final_date.getMinutes());
                myRef.child("p1").child("endDate").setValue(expiryDate);

                if (!add_card)
                {
                    myRef.child("p" + String.valueOf(i)).child("payCard").setValue(User.payMethod().get(cardId).first);
                }
                else {
                    String type = txt_new_type.getText().toString();
                    String num = txt_new_num.getText().toString();
                    User.payMethod().add(new Pair<>(num, type));
                    User.cardUpdateDB();
                    myRef.child("p" + String.valueOf(i)).child("payCard").setValue(User.payMethod().get(cardId).first);
                }

                //user update
                User.ongoing().get(0).set_prize(total_prize + User.ongoing().get(0).prize());
                User.ongoing().get(0).set_endDate(final_date);

                Intent intent = new Intent(ExtendDurationActivity.this, Favorite_parkingActivity.class);
                intent.putExtra("section", "ongoing");
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
package com.example.espark_1;

import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.AppCompatButton;
import androidx.appcompat.widget.AppCompatImageView;

import android.content.Intent;
import android.os.Bundle;
import android.util.Pair;
import android.view.View;
import android.widget.EditText;
import android.widget.RadioButton;
import android.widget.RadioGroup;

import java.util.Calendar;

public class PayCardActivity extends AppCompatActivity {
    private RadioGroup radioGroup;
    private RadioButton radioButton1;
    private RadioButton radioButton2;
    private RadioButton radioButton3;
    private AppCompatButton btn_next;
    private AppCompatImageView btn_pre;
    AppCompatImageView btn_add;
    EditText txt_new_type;
    EditText txt_new_num;
    Integer tot_time;
    int expiry_hour;
    int expiry_minute;
    int current_hour;
    int current_minute;
    float tot_prize;
    String name;
    int vehicleId;
    int cardId;
    boolean add_card = false;
    int calendarStart_Day;
    int calendarStart_Month;
    int calendarEnd_Day;
    int calendarEnd_Month;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_pay_card);

        cardId = -1;

        Bundle data = getIntent().getExtras();
        tot_time = (Integer) data.getInt("time");
        expiry_hour = data.getInt("hourExpiry");
        expiry_minute = data.getInt("minuteExpiry");
        tot_prize = data.getFloat("tot");
        name = data.getString("name");
        vehicleId = data.getInt("vehicle");
        current_hour = data.getInt("currentHour");
        current_minute = data.getInt("currentMinute");
        calendarStart_Day = data.getInt("calendarStart_Day");
        calendarStart_Month = data.getInt("calendarStart_Month") + 1;
        calendarEnd_Day = data.getInt("calendarEnd_Day");
        calendarEnd_Month = data.getInt("calendarEnd_Month") + 1;


        btn_add = findViewById(R.id.btn_add);
        txt_new_type= findViewById(R.id.ed_type);
        txt_new_num= findViewById(R.id.ed_num);
        txt_new_type.setVisibility(View.GONE);
        txt_new_num.setVisibility(View.GONE);

        btn_next = findViewById(R.id.btn_next);
        btn_next.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(PayCardActivity.this, SummaryParking.class);
                intent.putExtra("time", tot_time);
                intent.putExtra("hourExpiry", expiry_hour);
                intent.putExtra("minuteExpiry", expiry_minute);
                intent.putExtra("tot", tot_prize);
                intent.putExtra("name", name);
                intent.putExtra("vehicle", vehicleId);
                intent.putExtra("currentHour", current_hour);
                intent.putExtra("currentMinute", current_minute);
                intent.putExtra("calendarStart_Day", calendarStart_Day);
                intent.putExtra("calendarStart_Month", calendarStart_Month);
                intent.putExtra("calendarEnd_Day", calendarEnd_Day);
                intent.putExtra("calendarEnd_Month", calendarEnd_Month);
                if (!add_card)
                {
                    intent.putExtra("card", cardId);
                }
                else {
                    String type = txt_new_type.getText().toString();
                    String num = txt_new_num.getText().toString();
                    User.payMethod().add(new Pair<>(num, type));
                    User.cardUpdateDB();
                    intent.putExtra("card", User.payMethod().size() - 1);
                }
                startActivity(intent);
            }
        });

        if (cardId == -1)
        {
            btn_next.setEnabled(false);
        }

        btn_pre = findViewById(R.id.btn_cancel);
        btn_pre.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(PayCardActivity.this, HomepageActivity.class);
                startActivity(intent);
            }
        });

        radioGroup = findViewById(R.id.radioGroup);
        int iconResource = add_card ? R.drawable.ic_min : R.drawable.ic_add;
        btn_add.setImageResource(iconResource);
        // Get the RadioGroup from the layout

        // Create and add radio buttons dynamically
        //String[] options = {"Option 1", "Option 2", "Option 3"};
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
                    String selectedOption = selectedRadioButton.getText().toString();
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
    //disable back button
    @Override
    public void onBackPressed(){}
}
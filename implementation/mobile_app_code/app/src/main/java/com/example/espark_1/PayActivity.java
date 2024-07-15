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
import android.widget.TextView;

import java.util.Calendar;
import java.util.List;

public class PayActivity extends AppCompatActivity {

    AppCompatButton btn_minus;
    AppCompatButton btn_plus;
    AppCompatImageView btn_cancel;
    AppCompatButton btn_next;
    AppCompatImageView btn_add;
    EditText txt_new_brand;
    EditText txt_new_model;
    EditText txt_new_license;


    private String parkingID;
    private String parkingName;
    private Float parkingCostH;
    TextView txt_time;
    TextView txt_startH;
    TextView txt_endH;
    TextView txt_total;
    Integer tot_stop_time;
    Calendar calendar;
    Calendar calendarStart;
    int current_hour;
    int current_minute;
    int final_hour;
    int final_minute;
    float total_prize;
    int vehicle_id;
    private Parking myParking;
    boolean add_vehicle = false;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_pay);

        tot_stop_time = 0;
        total_prize = 0;
        vehicle_id = -1;

        Bundle data = getIntent().getExtras();
        //myParking = (Parking) data.getParcelable("parking");
        parkingID = data.getString("id");
        parkingName = data.getString("name");
        parkingCostH = data.getFloat("costH") / 2;  //cost per half hour

        txt_new_brand = findViewById(R.id.ed_brand);
        txt_new_model = findViewById(R.id.ed_model);
        txt_new_license = findViewById(R.id.ed_license);
        txt_new_brand.setVisibility(View.GONE);
        txt_new_model.setVisibility(View.GONE);
        txt_new_license.setVisibility(View.GONE);




        btn_cancel = findViewById(R.id.btn_cancel);
        btn_cancel.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                tot_stop_time = 0;
                total_prize = 0;
                Intent intent = new Intent(PayActivity.this, ParkingActivity.class);
                intent.putExtra("parking", parkingID);
                startActivity(intent);
            }
        });

        //next iff user select time and add vehicle details
        btn_next = findViewById(R.id.btn_next);
        btn_next.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(PayActivity.this, PayCardActivity.class);
                intent.putExtra("time", tot_stop_time);
                intent.putExtra("hourExpiry", final_hour);
                intent.putExtra("minuteExpiry", final_minute);
                intent.putExtra("tot", total_prize);
                intent.putExtra("name", parkingName);
                intent.putExtra("currentHour", current_hour);
                intent.putExtra("currentMinute", current_minute);
                intent.putExtra("calendarStart_Day", calendarStart.get(Calendar.DAY_OF_MONTH));
                intent.putExtra("calendarStart_Month", calendarStart.get(Calendar.MONTH));
                intent.putExtra("calendarEnd_Day", calendar.get(Calendar.DAY_OF_MONTH));
                intent.putExtra("calendarEnd_Month", calendar.get(Calendar.MONTH));
                if (!add_vehicle)
                {
                    intent.putExtra("vehicle", vehicle_id);
                }
                else {
                    String brand = txt_new_brand.getText().toString();
                    String model = txt_new_model.getText().toString();
                    String license = txt_new_license.getText().toString();
                    User.vehicle().add(new Pair<>(license, brand));
                    User.vehicleUpdateDB();
                    intent.putExtra("vehicle", User.vehicle().size() - 1);
                }
                startActivity(intent);
            }
        });

        if (vehicle_id == -1)
        {
            btn_next.setEnabled(false);
        }


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
                final_hour= calendar.get(Calendar.HOUR_OF_DAY);
                final_minute= calendar.get(Calendar.MINUTE);
                txt_endH.setText(String.valueOf(final_hour) + ":" + String.valueOf(final_minute));

                total_prize = total_prize - parkingCostH;
                txt_total.setText("€" + String.valueOf(total_prize));
            }
        });
        btn_plus = findViewById(R.id.btn_plus);
        btn_plus.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
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
                final_hour= calendar.get(Calendar.HOUR_OF_DAY);
                final_minute= calendar.get(Calendar.MINUTE);
                txt_endH.setText(String.valueOf(final_hour) + ":" + String.valueOf(final_minute));

                total_prize = total_prize + parkingCostH;
                txt_total.setText("€" + String.valueOf(total_prize));
            }
        });
        calendar = Calendar.getInstance();
        calendarStart = calendar;
        current_hour= calendar.get(Calendar.HOUR_OF_DAY);
        current_minute= calendar.get(Calendar.MINUTE);
        txt_startH = findViewById(R.id.txt_startH);
        txt_startH.setText(String.valueOf(current_hour) + ":" + String.valueOf(current_minute));

        txt_endH = findViewById(R.id.txt_endH);
        txt_endH.setText(String.valueOf(current_hour) + ":" + String.valueOf(current_minute));

        txt_total = findViewById(R.id.txt_total);
        txt_total.setText("€0.00");


        // Get the RadioGroup from the layout
        RadioGroup radioGroup = findViewById(R.id.radioGroup);
        btn_add = findViewById(R.id.btn_add);
        int iconResource = add_vehicle ? R.drawable.ic_min : R.drawable.ic_add;
        btn_add.setImageResource(iconResource);

        //String[] options = {"Option 1", "Option 2", "Option 3"};
        for (int i = 0; i < User.vehicle().size(); i++) {
            RadioButton radioButton = new RadioButton(this);
            radioButton.setId(i); // Set a unique ID for each radio button
            radioButton.setText("Model: " + User.vehicle().get(i).second + ", License plate:" + User.vehicle().get(i).first);
            radioGroup.addView(radioButton);
        }

        radioGroup.setOnCheckedChangeListener(new RadioGroup.OnCheckedChangeListener() {
            @Override
            public void onCheckedChanged(RadioGroup group, int checkedId) {

                    RadioButton selectedRadioButton = findViewById(checkedId);
                    if (selectedRadioButton != null) {
                        String selectedOption = selectedRadioButton.getText().toString();

                        vehicle_id = checkedId;
                        btn_next.setEnabled(true);
                        if (add_vehicle) {
                            add_vehicle = false;
                            btn_add.setImageResource(R.drawable.ic_add);
                            txt_new_brand.setVisibility(View.GONE);
                            txt_new_model.setVisibility(View.GONE);
                            txt_new_license.setVisibility(View.GONE);
                        }
                    }
            }
        });


        btn_add.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                add_vehicle = !add_vehicle;
                if (add_vehicle) {
                    radioGroup.clearCheck();
                    btn_add.setImageResource(R.drawable.ic_min);
                    txt_new_brand.setVisibility(View.VISIBLE);
                    txt_new_model.setVisibility(View.VISIBLE);
                    txt_new_license.setVisibility(View.VISIBLE);
                    btn_next.setEnabled(true);
                }
                else {
                    btn_add.setImageResource(R.drawable.ic_add);
                    txt_new_brand.setVisibility(View.GONE);
                    txt_new_model.setVisibility(View.GONE);
                    txt_new_license.setVisibility(View.GONE);
                    btn_next.setEnabled(false);
                }
            }
        });
    }
    //disable back button
    @Override
    public void onBackPressed(){}
}
package com.example.espark_1;

import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.AppCompatButton;
import androidx.appcompat.widget.AppCompatImageView;

import android.content.Intent;
import android.graphics.drawable.Drawable;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.TextView;

import com.google.android.material.slider.RangeSlider;
import com.google.android.material.slider.Slider;

public class FilterActivity extends AppCompatActivity {

    AppCompatButton btn_minus;
    AppCompatButton btn_plus;
    AppCompatImageView btn_cancel;
    AppCompatButton btn_next;
    AppCompatImageView btn_car;
    AppCompatImageView btn_motor;
    AppCompatImageView btn_charge;
    AppCompatImageView btn_yellow;
    AppCompatImageView btn_pink;
    TextView txt_prize;
    private Slider slider;
    boolean car = true;
    boolean motor = true;
    boolean charge = false;
    boolean yellow = false;
    boolean pink = false;
    Float max_prize = 5F;
    boolean filter_active = true;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_filter);


        Intent intent = getIntent();

        // Retrieve extras from the intent
        if (intent != null) {
            car = intent.getBooleanExtra("car", true);
            motor = intent.getBooleanExtra("motor", true);
            pink = intent.getBooleanExtra("pink", false);
            charge = intent.getBooleanExtra("charge", false);
            yellow = intent.getBooleanExtra("yellow", false);
            max_prize = intent.getFloatExtra("max_prize", 0F);
            if (max_prize.equals(-1F))
            {
                max_prize = 5F;
            }
        }

        //btn_minus = findViewById(R.id.btn_min);
       // btn_plus = findViewById(R.id.btn_plus);
        btn_cancel = findViewById(R.id.image_exit);
        btn_next = findViewById(R.id.btn_next);
        btn_car = findViewById(R.id.btn_car);
        btn_motor = findViewById(R.id.btn_motor);
        btn_charge = findViewById(R.id.icon_charge);
        btn_yellow = findViewById(R.id.icon_yellow);
        btn_pink = findViewById(R.id.icon_pink);
        txt_prize = findViewById(R.id.txt_maxprize);

        int iconResource = car ? R.drawable.border_check : R.drawable.border_notcheck;
        btn_car.setImageResource(iconResource);
        Log.d("CAR", String.valueOf(car));
        int iconResource1 = motor ? R.drawable.border_check : R.drawable.border_notcheck;
        btn_motor.setImageResource(iconResource1);
        Log.d("MOTOR", String.valueOf(motor));
        int iconResource2 = charge ? R.drawable.border_charging_selected : R.drawable.border_charging;
        btn_charge.setImageResource(iconResource2);
        int iconResource3 = yellow ? R.drawable.border_disabled_selected : R.drawable.border_disabled;
        btn_yellow.setImageResource(iconResource3);
        int iconResource4 = pink ? R.drawable.border_pink_selected : R.drawable.border_pink;
        btn_pink.setImageResource(iconResource4);

        btn_car.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                car = !car;
                int iconResource = car ? R.drawable.border_check : R.drawable.border_notcheck;
                btn_car.setImageResource(iconResource);
                /*if (motor)
                {
                    motor = false;
                    btn_motor.setImageResource(R.drawable.border_notcheck);
                }*/
            }
        });

        btn_motor.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                motor = !motor;
                int iconResource = motor ? R.drawable.border_check : R.drawable.border_notcheck;
                btn_motor.setImageResource(iconResource);
                /*if (car)
                {
                    car = false;
                    btn_car.setImageResource(R.drawable.border_notcheck);
                }*/
            }
        });

        btn_charge.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                charge = !charge;
                int iconResource = charge ? R.drawable.border_charging_selected : R.drawable.border_charging;
                btn_charge.setImageResource(iconResource);
                if (yellow)
                {
                    yellow = false;
                    btn_yellow.setImageResource(R.drawable.border_disabled);
                }
                if (pink)
                {
                    pink = false;
                    btn_pink.setImageResource(R.drawable.border_pink);
                }
            }
        });

        btn_yellow.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                yellow = !yellow;
                int iconResource = yellow ? R.drawable.border_disabled_selected : R.drawable.border_disabled;
                btn_yellow.setImageResource(iconResource);
                if (charge)
                {
                    charge = false;
                    btn_charge.setImageResource(R.drawable.border_charging);
                }
                if (pink)
                {
                    pink = false;
                    btn_pink.setImageResource(R.drawable.border_pink);
                }
            }
        });

        btn_pink.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                pink = !pink;
                int iconResource = pink ? R.drawable.border_pink_selected : R.drawable.border_pink;
                btn_pink.setImageResource(iconResource);
                if (charge)
                {
                    charge = false;
                    btn_charge.setImageResource(R.drawable.border_charging);
                }
                if (yellow)
                {
                    yellow = false;
                    btn_yellow.setImageResource(R.drawable.border_disabled);
                }
            }
        });

        btn_next.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                /*if (!car && !motor) {
                    car = true;
                    motor = true;
                }*/
                filter_active = true;
                //Passare i filtri selezionati
                Intent intent = new Intent(FilterActivity.this, HomepageActivity.class);
                intent.putExtra("car", car);
                intent.putExtra("motor", motor);
                intent.putExtra("charge", charge);
                intent.putExtra("pink", pink);
                intent.putExtra("yellow", yellow);
                intent.putExtra("max_prize", max_prize);
                intent.putExtra("filterActive", filter_active);
                startActivity(intent);
            }
        });
        btn_cancel.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                filter_active = false;
                Intent intent = new Intent(FilterActivity.this, HomepageActivity.class);
                intent.putExtra("filterActive", filter_active);
                startActivity(intent);
            }
        });


        slider = findViewById(R.id.slider);
        slider.setValue(max_prize);
        txt_prize.setText("€" + String.valueOf(max_prize) + " /hr");
        max_prize = slider.getValue();
        slider.setTrackActiveTintList(getResources().getColorStateList(R.color.white));
        slider.setTrackInactiveTintList(getResources().getColorStateList(R.color.grey));
        slider.setCustomThumbDrawable(getResources().getDrawable(R.drawable.slider));
        //slider.setThumbTintList(getResources().getDrawable(R.drawable.icon_car));
        slider.addOnChangeListener(new Slider.OnChangeListener() {
            @Override
            public void onValueChange(Slider s, float value, boolean fromUser) {

                max_prize = s.getValue();
                txt_prize.setText("€" + String.valueOf(s.getValue()) + " /hr");
                //Log.d("FILTER", String.valueOf(s.getValue()));
            }
        });


    }
    //disable back button
    @Override
    public void onBackPressed(){}
}
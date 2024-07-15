package com.example.espark_1;

import static com.example.espark_1.SECTION_P.selected_slot;

import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.AppCompatButton;
import androidx.appcompat.widget.AppCompatImageView;

import android.content.Intent;
import android.graphics.Color;
import android.graphics.Typeface;
import android.os.Bundle;
import android.util.Pair;
import android.view.Gravity;
import android.view.View;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.TextView;

import com.example.espark_1.databinding.ActivityVehicleBinding;

import com.google.firebase.database.FirebaseDatabase;

import java.text.SimpleDateFormat;
import java.util.Calendar;

public class VehicleActivity extends DrawerBaseActivity {
    LinearLayout parentLayout;
    EditText txt_new_brand;
    EditText txt_new_model;
    EditText txt_new_license;
    AppCompatImageView btn_add;
    boolean add_vehicle = false;
    AppCompatButton btn_next;
    AppCompatImageView btn_exit;
    private ActivityVehicleBinding binding;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        binding = ActivityVehicleBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());

        btn_next= findViewById(R.id.btn_next);
        txt_new_brand = findViewById(R.id.ed_brand);
        txt_new_model = findViewById(R.id.ed_model);
        txt_new_license = findViewById(R.id.ed_license);
        txt_new_brand.setVisibility(View.GONE);
        txt_new_model.setVisibility(View.GONE);
        txt_new_license.setVisibility(View.GONE);

        btn_exit = findViewById(R.id.image_exit);
        btn_exit.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(VehicleActivity.this, AccountActivity.class);
                startActivity(intent);
            }
        });

        createContent();
        btn_next.setEnabled(false);
        btn_next.setAlpha(0);
        btn_add = findViewById(R.id.btn_add);
        int iconResource = add_vehicle ? R.drawable.ic_min : R.drawable.ic_add;
        btn_add.setImageResource(iconResource);

        btn_add.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                add_vehicle = !add_vehicle;
                if (add_vehicle) {
                    btn_add.setImageResource(R.drawable.ic_min);
                    txt_new_brand.setVisibility(View.VISIBLE);
                    txt_new_model.setVisibility(View.VISIBLE);
                    txt_new_license.setVisibility(View.VISIBLE);
                    btn_next.setAlpha(1);

                    btn_next.setEnabled(true);
                }
                else {
                    btn_add.setImageResource(R.drawable.ic_add);
                    txt_new_brand.setVisibility(View.GONE);
                    txt_new_model.setVisibility(View.GONE);
                    txt_new_license.setVisibility(View.GONE);
                    btn_next.setAlpha(0);
                    btn_next.setEnabled(false);
                }
            }
        });

        btn_next.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(VehicleActivity.this, VehicleActivity.class);
                String brand = txt_new_brand.getText().toString();
                String model = txt_new_model.getText().toString();
                String license = txt_new_license.getText().toString();
                User.vehicle().add(new Pair<>(license, brand + model));
                User.vehicleUpdateDB();
                startActivity(intent);
            }
        });
    }


    private void createContent()
    {
        parentLayout = findViewById(R.id.parentlayout); // Assuming you have a parent layout in your XML
        int l = User.vehicle().size();

        for (int i = 0; i < l; i++) {
            LinearLayout roundedContainer = createRoundedContainer(i);
            // Add margin to the dynamically created container
            LinearLayout.LayoutParams layoutParams = new LinearLayout.LayoutParams(
                    LinearLayout.LayoutParams.MATCH_PARENT,
                    LinearLayout.LayoutParams.WRAP_CONTENT
            );
            layoutParams.setMargins(40, 40, 40, 60);
            roundedContainer.setLayoutParams(layoutParams);
            parentLayout.addView(roundedContainer);
        }
    }

    private LinearLayout createRoundedContainer(int i) {
        LinearLayout roundedContainer = new LinearLayout(this);
        roundedContainer.setLayoutParams(new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
        ));
        roundedContainer.setOrientation(LinearLayout.VERTICAL);
        roundedContainer.setBackgroundResource(R.drawable.container);
        roundedContainer.setElevation(getResources().getDimension(R.dimen.elevation_value));


        LinearLayout linearLayout1 = new LinearLayout(this);
        LinearLayout.LayoutParams layoutParams = new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
        );
        layoutParams.setMargins(0, 10, 0, 0);
        linearLayout1.setLayoutParams(layoutParams);
        linearLayout1.setOrientation(LinearLayout.VERTICAL);

        LinearLayout linearLayout2 = new LinearLayout(this);
        linearLayout2.setOrientation(LinearLayout.HORIZONTAL);
        linearLayout2.setLayoutParams(layoutParams);

        String name = User.vehicle().get(i).first;
        String prize = User.vehicle().get(i).second;



        TextView txt_name = new TextView(this);
        //textView12.setId(R.id.txt_wewe);
        LinearLayout.LayoutParams layoutParams1 = new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.WRAP_CONTENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
        );
        layoutParams1.setMargins(20, 0, 0, 0);
        txt_name.setText(name);
        txt_name.setTextColor(Color.parseColor("#000000"));
        txt_name.setTypeface(null, Typeface.BOLD);
        txt_name.setLayoutParams(layoutParams1);
        txt_name.setTextSize(18);

        View view = new View(this);
        view.setLayoutParams(new LinearLayout.LayoutParams(
                0,
                0,
                1));

        LinearLayout.LayoutParams layoutParams2 = new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.WRAP_CONTENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
        );
        layoutParams2.setMargins(0, 0, 20, 0);

        TextView txt_prize = new TextView(this);
        txt_prize.setText(prize);
        txt_prize.setTextColor(Color.parseColor("#000000"));
        txt_prize.setTextSize(15);
        txt_prize.setTypeface(null, Typeface.BOLD);
        txt_prize.setLayoutParams(layoutParams2);

        linearLayout2.addView(txt_name);
        linearLayout2.addView(view);
        linearLayout2.addView(txt_prize);


        linearLayout1.addView(linearLayout2);

        roundedContainer.addView(linearLayout1);


        return roundedContainer;
    }
    //disable back button
    @Override
    public void onBackPressed(){}
}
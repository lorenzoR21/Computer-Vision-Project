package com.example.espark_1;

import static android.content.ContentValues.TAG;

import static com.example.espark_1.SECTION_P.selected_slot;

import android.content.Intent;
import android.graphics.Color;
import android.graphics.Typeface;
import android.graphics.drawable.ColorDrawable;
import android.os.Bundle;
import android.util.Log;
import android.util.Pair;
import android.view.Gravity;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.TextView;

import androidx.appcompat.widget.AppCompatButton;
import androidx.appcompat.widget.AppCompatImageView;

import com.example.espark_1.databinding.ActivityFavoriteParkingBinding;
import com.google.firebase.database.DataSnapshot;
import com.google.firebase.database.DatabaseError;
import com.google.firebase.database.DatabaseReference;
import com.google.firebase.database.FirebaseDatabase;
import com.google.firebase.database.ValueEventListener;

import org.w3c.dom.Text;

import java.text.SimpleDateFormat;
import java.util.AbstractList;
import java.util.ArrayList;
import java.util.List;

enum SECTION_P {ongoing, completed, favourite, selected_slot}
public class Favorite_parkingActivity extends DrawerBaseActivity {

    private ActivityFavoriteParkingBinding binding;
    private FirebaseDatabase database;
    private DatabaseReference myRef;
    private ValueEventListener mValueEventListener;
    LinearLayout parentLayout;
    SECTION_P sp;
    AppCompatButton btn_f;
    AppCompatButton btn_c;
    AppCompatButton btn_o;
    AppCompatButton btn_sl;
    List<Pair<Pair<Pair<String, String>, String>, TextView>> slot_occupation;
    List<Pair<Pair<Pair<String, String>, String>, AppCompatImageView>> slot_delete;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        binding = ActivityFavoriteParkingBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());

        slot_occupation = new ArrayList<>();
        slot_delete = new ArrayList<>();

        int newBackgroundColor1 = Color.parseColor("#0DB665");
        ColorDrawable colorDrawableGreen = new ColorDrawable(newBackgroundColor1);
        int newBackgroundColor2 = Color.parseColor("#FFFFFF");
        ColorDrawable colorDrawableWhite = new ColorDrawable(newBackgroundColor2);

        Bundle data = getIntent().getExtras();
        sp = SECTION_P.valueOf(data.getString("section"));
        //Log.d("VALUE SP", String.valueOf(sp));

        btn_f = findViewById(R.id.btn_f);

        //default
        btn_f.setBackground(colorDrawableGreen);

        btn_f.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v)
            {
                sp = SECTION_P.favourite;
                btn_f.setBackground(getResources().getDrawable(R.drawable.border_selected));
                btn_c.setBackground(getResources().getDrawable(R.drawable.border));
                btn_o.setBackground(getResources().getDrawable(R.drawable.border));
                btn_sl.setBackground(getResources().getDrawable(R.drawable.border));

                btn_f.setTextColor(getResources().getColor(R.color.white));
                btn_c.setTextColor(getResources().getColor(R.color.blue));
                btn_o.setTextColor(getResources().getColor(R.color.blue));
                btn_sl.setTextColor(getResources().getColor(R.color.blue));
                parentLayout.removeAllViews();
                createContent();
            }
        });
        btn_c = findViewById(R.id.btn_c);
        btn_c.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v)
            {
                sp = SECTION_P.completed;
                btn_f.setBackground(getResources().getDrawable(R.drawable.border));
                btn_c.setBackground(getResources().getDrawable(R.drawable.border_selected));
                btn_o.setBackground(getResources().getDrawable(R.drawable.border));
                btn_sl.setBackground(getResources().getDrawable(R.drawable.border));

                btn_f.setTextColor(getResources().getColor(R.color.blue));
                btn_c.setTextColor(getResources().getColor(R.color.white));
                btn_o.setTextColor(getResources().getColor(R.color.blue));
                btn_sl.setTextColor(getResources().getColor(R.color.blue));
                parentLayout.removeAllViews();
                createContent();
            }
        });
        btn_o = findViewById(R.id.btn_o);
        btn_o.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v)
            {
                sp = SECTION_P.ongoing;
                btn_f.setBackground(getResources().getDrawable(R.drawable.border));
                btn_c.setBackground(getResources().getDrawable(R.drawable.border));
                btn_o.setBackground(getResources().getDrawable(R.drawable.border_selected));
                btn_sl.setBackground(getResources().getDrawable(R.drawable.border));

                btn_f.setTextColor(getResources().getColor(R.color.blue));
                btn_c.setTextColor(getResources().getColor(R.color.blue));
                btn_o.setTextColor(getResources().getColor(R.color.white));
                btn_sl.setTextColor(getResources().getColor(R.color.blue));
                parentLayout.removeAllViews();
                createContent();
            }
        });
        btn_sl = findViewById(R.id.btn_sl);
        btn_sl.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v)
            {
                slot_occupation.clear();
                slot_delete.clear();
                sp = selected_slot;
                btn_f.setBackground(getResources().getDrawable(R.drawable.border));
                btn_c.setBackground(getResources().getDrawable(R.drawable.border));
                btn_o.setBackground(getResources().getDrawable(R.drawable.border));
                btn_sl.setBackground(getResources().getDrawable(R.drawable.border_selected));

                btn_f.setTextColor(getResources().getColor(R.color.blue));
                btn_c.setTextColor(getResources().getColor(R.color.blue));
                btn_o.setTextColor(getResources().getColor(R.color.blue));
                btn_sl.setTextColor(getResources().getColor(R.color.white));
                parentLayout.removeAllViews();
                createContent();

                for (int i = 0; i < User.selectedSlot().size(); i++)
                {
                    update_slot(i);
                }

            }
        });

        createContent();

        if(sp == SECTION_P.favourite)
        {
            btn_f.setBackground(getResources().getDrawable(R.drawable.border_selected));
            btn_c.setBackground(getResources().getDrawable(R.drawable.border));
            btn_o.setBackground(getResources().getDrawable(R.drawable.border));
            btn_sl.setBackground(getResources().getDrawable(R.drawable.border));

            btn_f.setTextColor(getResources().getColor(R.color.white));
            btn_c.setTextColor(getResources().getColor(R.color.blue));
            btn_o.setTextColor(getResources().getColor(R.color.blue));
            btn_sl.setTextColor(getResources().getColor(R.color.blue));
            parentLayout.removeAllViews();
            createContent();
        }
        if(sp == SECTION_P.completed)
        {
            btn_f.setBackground(getResources().getDrawable(R.drawable.border));
            btn_c.setBackground(getResources().getDrawable(R.drawable.border_selected));
            btn_o.setBackground(getResources().getDrawable(R.drawable.border));
            btn_sl.setBackground(getResources().getDrawable(R.drawable.border));

            btn_f.setTextColor(getResources().getColor(R.color.blue));
            btn_c.setTextColor(getResources().getColor(R.color.white));
            btn_o.setTextColor(getResources().getColor(R.color.blue));
            btn_sl.setTextColor(getResources().getColor(R.color.blue));
            parentLayout.removeAllViews();
            createContent();
        }
        if(sp == SECTION_P.ongoing)
        {
            btn_f.setBackground(getResources().getDrawable(R.drawable.border));
            btn_c.setBackground(getResources().getDrawable(R.drawable.border));
            btn_o.setBackground(getResources().getDrawable(R.drawable.border_selected));
            btn_sl.setBackground(getResources().getDrawable(R.drawable.border));

            btn_f.setTextColor(getResources().getColor(R.color.blue));
            btn_c.setTextColor(getResources().getColor(R.color.blue));
            btn_o.setTextColor(getResources().getColor(R.color.white));
            btn_sl.setTextColor(getResources().getColor(R.color.blue));
            parentLayout.removeAllViews();
            createContent();
        }
        if(sp == selected_slot)
        {
            slot_occupation.clear();
            slot_delete.clear();
            //Log.d("ABCD", "CIAO BELLI");
            btn_f.setBackground(getResources().getDrawable(R.drawable.border));
            btn_c.setBackground(getResources().getDrawable(R.drawable.border));
            btn_o.setBackground(getResources().getDrawable(R.drawable.border));
            btn_sl.setBackground(getResources().getDrawable(R.drawable.border_selected));

            btn_f.setTextColor(getResources().getColor(R.color.blue));
            btn_c.setTextColor(getResources().getColor(R.color.blue));
            btn_o.setTextColor(getResources().getColor(R.color.blue));
            btn_sl.setTextColor(getResources().getColor(R.color.white));
            parentLayout.removeAllViews();
            createContent();

            Log.d("ABC", String.valueOf(User.selectedSlot().size()));
            for (int i = 0; i < User.selectedSlot().size(); i++)
            {
                update_slot(i);
            }
        }

    }

    private void createContent()
    {
        parentLayout = findViewById(R.id.parentlayout); // Assuming you have a parent layout in your XML
        int l = 0;
        if(sp == SECTION_P.favourite)
        {
            l = User.favouriteParking().size();
        }
        if(sp == SECTION_P.completed)
        {
            l = User.history().size();
        }
        if(sp == SECTION_P.ongoing)
        {
            if (User.ongoing().size() == 0)
            {
                l = 0;
            }
            else {
                if (User.ongoing().get(0).ongoing().equals("true")) {
                    l = User.ongoing().size();
                } else if (User.ongoing().get(0).ongoing().equals("false")) {
                    l = 0;
                }
            }
        }
        if(sp == selected_slot)
        {
            l = User.selectedSlot().size();
        }

        for (int i = 0; i < l; i++) {
            LinearLayout roundedContainer = createRoundedContainer(i);
            // Add margin to the dynamically created container
            LinearLayout.LayoutParams layoutParams = new LinearLayout.LayoutParams(
                    LinearLayout.LayoutParams.MATCH_PARENT,
                    getResources().getDimensionPixelSize(R.dimen.container_height)
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
                getResources().getDimensionPixelSize(R.dimen.container_height)
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

        AppCompatImageView img_star = new AppCompatImageView(this);
        //img_star.setId(R.id.favorite_icon);
        LinearLayout.LayoutParams layoutParams2 = new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.WRAP_CONTENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
        );
        layoutParams2.setMargins(0, 0, 20, 0);
        img_star.setLayoutParams(layoutParams2);
        //img_star.setImageResource(R.drawable.icon_star_black);

        String name = "";
        String startDate = "";
        String endDate = "";
        String prize = "";
        String buttonText = "Parking Page";

        TextView txt_occupation = new TextView(this);
        LinearLayout.LayoutParams layoutParamsOCC = new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.WRAP_CONTENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
        );
        layoutParamsOCC.setMargins(20, 0, 0, 0);
        txt_occupation.setText("");
        txt_occupation.setTextColor(Color.parseColor("#000000"));
        txt_occupation.setLayoutParams(layoutParamsOCC);
        txt_occupation.setTextSize(15);
        //txt_occupation.setPadding(30, 16, 0, 0);

        // Define the text

        if(sp == SECTION_P.favourite)
        {
            name = User.favouriteParking().get(i).second;
            img_star.setImageResource(R.drawable.icon_star_black);
        }
        if(sp == SECTION_P.completed)
        {
            SimpleDateFormat format=new SimpleDateFormat("dd-MM-yyyy HH:mm");
            name = User.history().get(i).name();
            startDate = "Start date: " + format.format(User.history().get(i).startDate());
            endDate = "End date: " + format.format(User.history().get(i).endDate());
            prize = "Total paid: €" + String.valueOf(User.history().get(i).prize());
        }
        if(sp == SECTION_P.ongoing)
        {
            SimpleDateFormat format=new SimpleDateFormat("dd-MM-yyyy HH:mm");
            name = User.ongoing().get(i).name();
            startDate = "Start date: " + format.format(User.ongoing().get(i).startDate());
            endDate = "End date: " + format.format(User.ongoing().get(i).endDate());
            prize = "Total paid: €" + String.valueOf(User.ongoing().get(i).prize());
            buttonText = "Extend the duration of the parking";
        }
        if(sp == selected_slot)
        {
            name = User.selectedSlot().get(i).first.first;
            startDate = "Selected slot number: " + User.selectedSlot().get(i).first.second;  // slotNum
            Pair<Pair<String, String>, String> s1 = User.selectedSlot().get(i);
            img_star.setImageResource(R.drawable.icon_trash);
            img_star.setTag(User.selectedSlot().get(i).first.second + User.selectedSlot().get(i).second);
            img_star.setOnClickListener(new View.OnClickListener() {
                @Override
                public void onClick(View v) {
                    database = FirebaseDatabase.getInstance();
                    myRef = database.getReference("user").child(User.id()).child("selectedSlot");
                        for (int j = 0; j < slot_delete.size(); j++)
                        {
                            if (img_star.getTag().equals(slot_delete.get(j).first.first.second + slot_delete.get(j).first.second))
                            {
                                User.selectedSlot().remove(slot_delete.get(j).first);
                                slot_delete.remove(j);
                                User.slotUpdateDB();
                                //createContent();
                                Intent intent = new Intent(Favorite_parkingActivity.this, Favorite_parkingActivity.class);
                                intent.putExtra("section", "selected_slot");
                                startActivity(intent);
                            }
                        }
                }
            });
            Pair<Pair<Pair<String, String>, String>, AppCompatImageView> s3 = new Pair<>(s1, img_star);

            txt_occupation.setText("");
            txt_occupation.setTextSize(getResources().getDimensionPixelSize(R.dimen.text_size));

            Pair<Pair<Pair<String, String>, String>, TextView> s2 = new Pair<>(s1, txt_occupation);

            slot_occupation.add(s2);
            slot_delete.add(s3);
        }


        /*descriptionTextView.setText(name + "\n" + prize + "\n" + startDate + "\n" + endDate);

        descriptionTextView.setTextSize(getResources().getDimensionPixelSize(R.dimen.text_size));
        descriptionTextView.setGravity(android.view.Gravity.CENTER_HORIZONTAL);*/

        AppCompatButton btn_nextPage = new AppCompatButton(this);
        //button.setId(R.id.btn);
        LinearLayout.LayoutParams layoutParams6 = new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.WRAP_CONTENT,
                100
        );
        //layoutParams6.setMargins(0, 25, 0, 0);
        btn_nextPage.setLayoutParams(layoutParams6);
        //btn_nextPage.setGravity(Gravity.CENTER);
        btn_nextPage.setBackgroundDrawable(getResources().getDrawable(R.drawable.btn_background_2));
        btn_nextPage.setTextColor(Color.parseColor("#FFFFFF"));
        btn_nextPage.setText(buttonText);
        btn_nextPage.setTextSize(12);
        /*Button button = new Button(this);
        button.setLayoutParams(new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.WRAP_CONTENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
        ));
        button.setText(buttonText);
        button.setGravity(android.view.Gravity.CENTER_HORIZONTAL);
        button.setPadding(
                getResources().getDimensionPixelSize(R.dimen.button_padding_start),
                0,
                getResources().getDimensionPixelSize(R.dimen.button_padding_end),
                0
        );*/

        if(sp == SECTION_P.favourite)
        {
            btn_nextPage.setOnClickListener(new View.OnClickListener() {
                @Override
                public void onClick(View v) {
                    Intent intent = new Intent(Favorite_parkingActivity.this, ParkingActivity.class);
                    intent.putExtra("parking", User.favouriteParking().get(i).first);
                    startActivity(intent);
                }
            });
        }
        if(sp == SECTION_P.completed)
        {
            btn_nextPage.setOnClickListener(new View.OnClickListener() {
                @Override
                public void onClick(View v) {
                    Intent intent = new Intent(Favorite_parkingActivity.this, ParkingActivity.class);
                    intent.putExtra("parking", User.history().get(i).id());
                    startActivity(intent);
                }
            });
        }
        if(sp == SECTION_P.ongoing)
        {
            btn_nextPage.setOnClickListener(new View.OnClickListener() {
                @Override
                public void onClick(View v) {
                    Intent intent = new Intent(Favorite_parkingActivity.this, ExtendDurationActivity.class);
                    startActivity(intent);
                }
            });
        }
        if(sp == selected_slot)
        {
            btn_nextPage.setOnClickListener(new View.OnClickListener() {
                @Override
                public void onClick(View v) {
                    Intent intent = new Intent(Favorite_parkingActivity.this, ParkingActivity.class);
                    intent.putExtra("parking", User.selectedSlot().get(i).second);
                    startActivity(intent);
                }
            });
        }


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


        linearLayout2.addView(txt_name);
        linearLayout2.addView(view);
        linearLayout2.addView(img_star);

        TextView txt_startDate = new TextView(this);
        //textView13.setId(R.id.txt_13);
        LinearLayout.LayoutParams layoutParams3 = new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.WRAP_CONTENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
        );
        layoutParams3.setMargins(20, 0, 0, 0);
        txt_startDate.setText(startDate);
        txt_startDate.setTextColor(Color.parseColor("#000000"));
        txt_startDate.setLayoutParams(layoutParams3);
        txt_startDate.setTextSize(15);

        TextView txt_endDate = new TextView(this);
        //textView14.setId(R.id.txt_14);
        LinearLayout.LayoutParams layoutParams4 = new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.WRAP_CONTENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
        );
        layoutParams4.setMargins(20, 0, 0, 0);
        txt_endDate.setText(endDate);
        txt_endDate.setTextColor(Color.parseColor("#000000"));
        txt_endDate.setLayoutParams(layoutParams4);
        txt_endDate.setTextSize(15);

        LinearLayout linearLayoutPA = new LinearLayout(this);
        linearLayoutPA.setOrientation(LinearLayout.HORIZONTAL);
        TextView txt_prize = new TextView(this);
        //textView15.setId(R.id.txt_15);
        LinearLayout.LayoutParams layoutParams5 = new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.WRAP_CONTENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
        );
        //layoutParams5.setMargins(0, 0, 20, 0);
        //txt_prizeORavailable.setGravity(Gravity.END);
        txt_prize.setText(prize);
        txt_prize.setTextColor(Color.parseColor("#000000"));
        //txt_prizeORavailable.setLayoutParams(layoutParams5);
        txt_prize.setTextSize(15);
        LinearLayout.LayoutParams layoutParams8 = new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
        );
        layoutParams8.setMargins(0, 0, 20, 0);
        linearLayoutPA.setLayoutParams(layoutParams8);
        linearLayoutPA.addView(txt_prize);
        linearLayoutPA.setGravity(Gravity.END);


        LinearLayout linearLayoutBTN = new LinearLayout(this);
        linearLayoutBTN.setOrientation(LinearLayout.HORIZONTAL);
        LinearLayout.LayoutParams layoutParams7 = new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
        );
        layoutParams7.setMargins(0, 25, 0, 0);
        linearLayoutBTN.setLayoutParams(layoutParams7);
        linearLayoutBTN.addView(btn_nextPage);
        linearLayoutBTN.setGravity(Gravity.CENTER);


        linearLayout1.addView(linearLayout2);
        linearLayout1.addView(txt_startDate);
        if(sp == selected_slot)
        {
            linearLayout1.addView(txt_occupation);
        }
        else {
            linearLayout1.addView(txt_endDate);
        }
        linearLayout1.addView(linearLayoutPA);
        linearLayout1.addView(linearLayoutBTN);

        //roundedContainer.addView(descriptionTextView);
        //roundedContainer.addView(button);
        roundedContainer.addView(linearLayout1);


        return roundedContainer;
    }

    void update_slot(int i)
    {

        database = FirebaseDatabase.getInstance();
        myRef = database.getReference("parking");
        Log.d("1", "WE");
        mValueEventListener = new ValueEventListener() {
            @Override
            public void onDataChange(DataSnapshot dataSnapshot) {
                DataSnapshot childSnapshot = dataSnapshot.child(slot_occupation.get(i).first.second);
                Log.d("2", slot_occupation.get(i).first.second);
                Integer status = -1;
                List<Parking_slot> slot_app = new ArrayList<>();
                for (DataSnapshot slotSnapshot : childSnapshot.child("slot").getChildren())
                {
                    String a = slotSnapshot.getKey();

                    String value = slotSnapshot.child("value").getValue(String.class);
                    Log.d("3", value);
                    Log.d("4", slot_occupation.get(i).first.first.second);
                    if (value.equals(slot_occupation.get(i).first.first.second))
                    {
                        status = slotSnapshot.child("status").getValue(Integer.class);
                        Log.d("5", String.valueOf(status));
                        break;

                    }

                }

                if (status.equals(0))
                {
                    slot_occupation.get(i).second.setText("available");
                    slot_occupation.get(i).second.setTextColor(getResources().getColor(R.color.green));
                }
                else if (status.equals(1))
                {
                    slot_occupation.get(i).second.setText("not available");
                    slot_occupation.get(i).second.setTextColor(getResources().getColor(R.color.red));
                }

            }

            @Override
            public void onCancelled(DatabaseError databaseError) {
                Log.e(TAG, "onCancelled", databaseError.toException());
            }
        };

        myRef.addValueEventListener(mValueEventListener);
    }

    //disable back button
    @Override
    public void onBackPressed(){}
}
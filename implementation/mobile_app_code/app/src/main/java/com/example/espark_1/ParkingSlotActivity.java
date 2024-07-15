package com.example.espark_1;

import static android.content.ContentValues.TAG;

import android.content.Intent;
import android.graphics.drawable.Drawable;
import android.os.Bundle;
import android.util.Pair;
import android.view.Gravity;
import android.view.View;
import android.view.ViewGroup;
import android.widget.LinearLayout;
import android.widget.TextView;
import androidx.appcompat.widget.AppCompatButton;
import androidx.appcompat.widget.AppCompatImageView;

import android.graphics.Color;
import android.os.Handler;
import android.util.Log;
import android.widget.Button;

import com.example.espark_1.databinding.ActivityParkingSlotBinding;
import com.google.firebase.database.DataSnapshot;
import com.google.firebase.database.DatabaseError;
import com.google.firebase.database.DatabaseReference;
import com.google.firebase.database.FirebaseDatabase;
import com.google.firebase.database.Query;
import com.google.firebase.database.ValueEventListener;

import org.mapsforge.map.rendertheme.renderinstruction.Line;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

public class ParkingSlotActivity extends DrawerBaseActivity {

    private ActivityParkingSlotBinding binding;
    private Parking myParking;
    private String parkingID;
    private String parkingName;
    private List<Parking_slot> parkingSlot;
    private LinearLayout layout_left;
    private LinearLayout layout_right;

    TextView txvalue;
    Handler handler = new Handler();
    boolean statusdevice = true;
    private AppCompatButton slot1;
    private AppCompatButton slot2;

    private int current_status_1 = 0;
    private int current_status_2 = 0;
    private int v1 = 0;
    private int v2 = 0;
    AppCompatImageView btn_canc;
    //public static String ip_address;

    private FirebaseDatabase database;
    private DatabaseReference myRef;
    private ValueEventListener mValueEventListener;

    private LinearLayout parkingContainerLeft;
    private LinearLayout parkingContainerRight;
    private List<Button> slotButton;
    TextView num_slot;
    TextView available_slot;
    TextView txt_line;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        binding = ActivityParkingSlotBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());

        Bundle data = getIntent().getExtras();
        //myParking = (Parking) data.getParcelable("parking");
        parkingID = data.getString("id");
        parkingName = data.getString("name");
        parkingSlot = (List<Parking_slot>) getIntent().getSerializableExtra("slot");



        TextView txt_parkingName = findViewById(R.id.name);
        txt_parkingName.setText(parkingName);
        txt_line = findViewById(R.id.line);
        txt_line.setText("");
        layout_left = findViewById(R.id.parking_layout1);
        layout_right = findViewById(R.id.parking_layout2);

        num_slot = findViewById(R.id.num_slot);
        available_slot = findViewById(R.id.txt_slotAvailable);
        num_slot.setText(String.valueOf(parkingSlot.size()));
        btn_canc = findViewById(R.id.image_exit);

        //txvalue = (TextView ) findViewById(R.id.tx_value);
        slotButton = new ArrayList<>();
        slotGenerator();
        statusChange();

        boolean w = false;
        boolean b = false;
        boolean g = false;
        boolean p = false;
        boolean y = false;
        for (int i = 0; i < parkingSlot.size(); i++)
        {
            if (parkingSlot.get(i).type() == TYPE_LINE.white)
                w = true;
            if (parkingSlot.get(i).type() == TYPE_LINE.blue)
                b = true;
            if (parkingSlot.get(i).type() == TYPE_LINE.pink)
                p = true;
            if (parkingSlot.get(i).type() == TYPE_LINE.yellow)
                y = true;
            if (parkingSlot.get(i).type() == TYPE_LINE.green)
                g = true;
        }
        if (w)
            txt_line.setText(txt_line.getText() + " White line: free slots");
        if (b)
            txt_line.setText(txt_line.getText() + " Blue line: paid slots");
        if (p)
            txt_line.setText(txt_line.getText() + " Pink line: pregnant slots");
        if (y)
            txt_line.setText(txt_line.getText() + " Yellow line: disabled");
        if (g)
            txt_line.setText(txt_line.getText() + " Green line: charge slots");
        //handler.postDelayed(status_data,0);

        btn_canc.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(ParkingSlotActivity.this, ParkingActivity.class);
                intent.putExtra("parking", parkingID);
                startActivity(intent);
            }
        });

    }

    public void statusChange()
    {
        database = FirebaseDatabase.getInstance();
        myRef = database.getReference("parking").child(parkingID).child("slot");
        Query query = myRef.orderByKey();


        mValueEventListener = new ValueEventListener() {
            @Override
            public void onDataChange(DataSnapshot dataSnapshot) {

                List<Parking_slot> slot_app = new ArrayList<>();
                for (DataSnapshot childSnapshot : dataSnapshot.getChildren())
                {

                    String a = childSnapshot.getKey();
                    String value = childSnapshot.child("value").getValue(String.class);
                    TYPE_LINE type_s = TYPE_LINE.valueOf(childSnapshot.child("type").getValue(String.class));
                    Integer status = childSnapshot.child("status").getValue(Integer.class);
                    Parking_slot app = new Parking_slot(a, value, type_s, status);
                    slot_app.add(app);

                    Comparator<Parking_slot> customComparator = (o1, o2) -> {
                        String s1 = o1.id();
                        String s2 = o2.id();

                        // Extract the numeric part and compare as integers
                        int num1 = Integer.parseInt(s1.substring(1));
                        int num2 = Integer.parseInt(s2.substring(1));

                        return Integer.compare(num1, num2);
                    };
                    Collections.sort(slot_app, customComparator);
                    //Log.d(TAG,"slotID: " + slotID + ", status: " + status);
                }
                for (int i = 0; i < slot_app.size(); i++)
                {
                    if (slot_app.get(i).status() == 0)
                    {
                        slotButton.get(i).setEnabled(true);
                        //slotButton.get(i).setBackgroundColor(Color.GREEN);
                        slotButton.get(i).setCompoundDrawablesWithIntrinsicBounds(
                                null,   // Left drawable
                                null, // Top drawable
                                null,   // Right drawable
                                null    // Bottom drawable
                        );

                    }
                    else {
                        slotButton.get(i).setEnabled(false);
                        //slotButton.get(i).setBackgroundColor(Color.RED);
                        slotButton.get(i).setCompoundDrawablesWithIntrinsicBounds(
                                getResources().getDrawable(R.drawable.icon_car),   // Left drawable
                                null, // Top drawable
                                null,   // Right drawable
                                null    // Bottom drawable
                        );
                    }



                }

                Integer notAvSlot = 0;
                for (int j = 0; j < slot_app.size(); j++)
                {
                    if (slot_app.get(j).status() == 1)
                    {
                        notAvSlot++;
                    }
                }

                if (slot_app.size() - notAvSlot > 0) //there are available slot
                {
                    available_slot.setTextColor(getResources().getColor(R.color.green));
                    available_slot.setText("Slot Available, parking occupation: " + String.valueOf(notAvSlot) + "/" + String.valueOf(slot_app.size()));
                }
                else
                {
                    available_slot.setTextColor(getResources().getColor(R.color.red));
                    available_slot.setText("Parking full");
                }
            }

            @Override
            public void onCancelled(DatabaseError databaseError) {
                Log.e(TAG, "onCancelled", databaseError.toException());
            }
        };

        query.addValueEventListener(mValueEventListener);
    }
    //disable back button
    @Override
    public void onBackPressed(){}
    public void slotGenerator()
    {
        //ViewGroup parkingLayout = findViewById(R.id.parking_layout);

        int numParkingSlots = parkingSlot.size();

        for (int i = 0; i < numParkingSlots; i++)
        {
            Button button = new Button(this);
            button.setLayoutParams(new LinearLayout.LayoutParams(
                    ViewGroup.LayoutParams.MATCH_PARENT,
                    180
            ));
            button.setText(parkingSlot.get(i).value());
            button.setGravity(Gravity.END | Gravity.TOP);
            TYPE_LINE color = parkingSlot.get(i).type();
            if (color == TYPE_LINE.white)
                button.setBackground(getResources().getDrawable(R.drawable.top_border_white));
            if (color == TYPE_LINE.blue)
                button.setBackground(getResources().getDrawable(R.drawable.top_border));
            if (color == TYPE_LINE.pink)
                button.setBackground(getResources().getDrawable(R.drawable.top_border_pink));
            if (color == TYPE_LINE.yellow)
                button.setBackground(getResources().getDrawable(R.drawable.top_border_yellow));
            if (color == TYPE_LINE.green)
                button.setBackground(getResources().getDrawable(R.drawable.top_border_green));
            //button.setBackgroundColor(Color.GREEN);
            button.setPadding(190, 20, 20, 20);

            button.setOnClickListener(new View.OnClickListener() {
                @Override
                public void onClick(View v) {

                    Pair<String, String> q = new Pair<>(parkingName, String.valueOf(button.getText()));
                    if (!User.selectedSlot().contains(new Pair<Pair<String, String>, String> (q, parkingID)))
                    {
                        User.selectedSlot().add(new Pair<Pair<String, String>, String> (q, parkingID));

                        database = FirebaseDatabase.getInstance();
                        myRef = database.getReference("user").child(User.id()).child("selectedSlot");
                        myRef.addListenerForSingleValueEvent(new ValueEventListener() {
                            @Override
                            public void onDataChange(DataSnapshot dataSnapshot) {
                                int i = 1;
                                for (DataSnapshot childSnapshot : dataSnapshot.getChildren())
                                {
                                    String id = childSnapshot.getKey();
                                    i++;
                                }

                                myRef.child("s" + String.valueOf(i)).child("parkingName").setValue(parkingName);
                                myRef.child("s" + String.valueOf(i)).child("slotNum").setValue(String.valueOf(button.getText()));
                                myRef.child("s" + String.valueOf(i)).child("parkingId").setValue(parkingID);
                            }

                            @Override
                            public void onCancelled(DatabaseError databaseError) {
                                Log.e(TAG, "onCancelled", databaseError.toException());
                            }
                        });
                    }



                    Intent intent = new Intent(ParkingSlotActivity.this, Favorite_parkingActivity.class);
                    intent.putExtra("section", "selected_slot");
                    startActivity(intent);
                }
            });



            slotButton.add(button);

            if (i % 2 == 0)
            {
                layout_left.addView(button);
            }
            else {
                layout_right.addView(button);
            }
        }
    }
}
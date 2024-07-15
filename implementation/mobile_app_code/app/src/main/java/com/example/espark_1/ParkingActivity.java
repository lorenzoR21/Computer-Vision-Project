package com.example.espark_1;


import static android.content.ContentValues.TAG;

import androidx.annotation.NonNull;
import androidx.appcompat.widget.AppCompatButton;
import androidx.appcompat.widget.AppCompatImageView;

import android.app.Dialog;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.graphics.drawable.ColorDrawable;
import android.os.Bundle;
import android.util.Log;
import android.util.Pair;
import android.view.View;
import android.view.ViewGroup;
import android.view.Window;
import android.widget.ImageView;
import android.widget.TextView;

import com.example.espark_1.databinding.ActivityParkingBinding;
import com.google.android.gms.tasks.OnFailureListener;
import com.google.android.gms.tasks.OnSuccessListener;
import com.google.firebase.database.DataSnapshot;
import com.google.firebase.database.DatabaseError;
import com.google.firebase.database.DatabaseReference;
import com.google.firebase.database.FirebaseDatabase;
import com.google.firebase.database.ValueEventListener;
import com.google.firebase.storage.FirebaseStorage;
import com.google.firebase.storage.StorageReference;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

public class ParkingActivity extends DrawerBaseActivity {

    private ActivityParkingBinding binding;
    private AppCompatImageView favoriteIcon;
    private boolean isFavorite = false;
    private String _parkingID;

    public Parking parking;

    private FirebaseDatabase database;
    private DatabaseReference myRef;
    private ValueEventListener mValueEventListener;
    TextView num_slot;
    TextView txt_prize;
    TextView txt_perhour;
    TextView txt_street;

    TextView available_slot;
    AppCompatButton btn_slot;
    AppCompatButton btn_pay;

    AppCompatImageView img_car;
    AppCompatImageView img_motor;
    AppCompatImageView img_yellow;
    AppCompatImageView img_pink;
    AppCompatImageView img_charge;
    AppCompatImageView exit;
    AppCompatImageView info;

    private ImageView imageParking;

    private FirebaseStorage storage = FirebaseStorage.getInstance();
    private StorageReference storageRef;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        binding = ActivityParkingBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());

       /* FirebaseDatabase database = FirebaseDatabase.getInstance();
        DatabaseReference myRef = database.getReference("name");*/

        Bundle data = getIntent().getExtras();
        _parkingID = (String) data.get("parking");


        img_car = findViewById(R.id.image_car) ;
        img_motor = findViewById(R.id.image_motor);
        img_yellow = findViewById(R.id.icon_yellow);
        img_pink = findViewById(R.id.icon_pink);
        img_charge = findViewById(R.id.icon_charge);
        txt_street= findViewById(R.id.txt_street);
        info = findViewById(R.id.info_icon);
        info.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                // Create a custom dialog
                final Dialog dialog = new Dialog(ParkingActivity.this);
                dialog.setContentView(R.layout.popup_info);

                // Initialize views inside the dialog
                TextView textViewInfo = dialog.findViewById(R.id.name1);
                AppCompatButton buttonClose = dialog.findViewById(R.id.button_close);

                Window window = dialog.getWindow();
                if (window != null) {
                    window.setLayout(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT);
                }

                buttonClose.setOnClickListener(new View.OnClickListener() {
                    @Override
                    public void onClick(View v) {
                        // Dismiss the dialog when the Close button is clicked
                        dialog.dismiss();
                    }
                });

                // Show the dialog
                dialog.show();
            }
        });
        for (int i = 0; i < User.favouriteParking().size(); i++)
        {
            if (User.favouriteParking().get(i).first.equals(_parkingID))
            {
                isFavorite = true;
                break;
            }
        }

        readParking();

        exit = findViewById(R.id.image_exit);
        exit.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(ParkingActivity.this, HomepageActivity.class);
                startActivity(intent);
            }
        });

        btn_pay = findViewById(R.id.button_pay);
        btn_pay.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(ParkingActivity.this, PayActivity.class);
                intent.putExtra("id", parking.id());
                intent.putExtra("name", parking.name());
                intent.putExtra("costH", parking.costHourly());
                startActivity(intent);
            }
        });

        btn_slot = findViewById(R.id.button_slot);
        btn_slot.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(ParkingActivity.this, ParkingSlotActivity.class);
                intent.putExtra("id", parking.id());
                intent.putExtra("name", parking.name());
                intent.putExtra("slot", (Serializable) parking.slot());
                startActivity(intent);
            }
        });

        available_slot = findViewById(R.id.txt_slotAvailable);

        favoriteIcon = findViewById(R.id.favorite_icon);
        favoriteIcon.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                isFavorite = !isFavorite;
                updateFavorite();
            }
        });

        int iconResource = isFavorite ? R.drawable.icon_star_black : R.drawable.icon_star;
        favoriteIcon.setImageResource(iconResource);

        imageParking = findViewById(R.id.imageParking);

        storageRef = storage.getReference().child("parkingImage/" + _parkingID + ".jpg");
        loadImage();

    }


    private void loadImage() {
        storageRef.getBytes(Long.MAX_VALUE)
                .addOnSuccessListener(new OnSuccessListener<byte[]>() {
                    @Override
                    public void onSuccess(byte[] bytes) {
                        // Convert the byte array to a Bitmap and set it to the ImageView
                        Bitmap bitmap = BitmapFactory.decodeByteArray(bytes, 0, bytes.length);
                        imageParking.setImageBitmap(bitmap);
                    }
                })
                .addOnFailureListener(new OnFailureListener() {
                    @Override
                    public void onFailure(@NonNull Exception exception) {
                        // Handle any errors
                    }
                });
    }

    // You can call this method whenever you want to change the image
    private void changeImage(String newImagePath) {
        storageRef = storage.getReference().child(newImagePath); // Replace with the new image's path
        loadImage();
    }


private void updateFavorite() {

        database = FirebaseDatabase.getInstance();
        myRef = database.getReference("user").child(User.id()).child("favouriteParking");
        if (isFavorite)
        {
            User.favouriteParking().add(new Pair<>(_parkingID, parking.name()));
            myRef.addListenerForSingleValueEvent(new ValueEventListener() {
                @Override
                public void onDataChange(DataSnapshot dataSnapshot) {

                    myRef.child(_parkingID).child("name").setValue(parking.name());
                }

                @Override
                public void onCancelled(DatabaseError databaseError) {
                    Log.e(TAG, "onCancelled", databaseError.toException());
                }
            });
        }
        else {
            User.favouriteParking().remove(new Pair<>(_parkingID, parking.name()));
            myRef.child(_parkingID).removeValue();
        }
        int iconResource = isFavorite ? R.drawable.icon_star_black : R.drawable.icon_star;
        favoriteIcon.setImageResource(iconResource);
    }

    public void readParking()
    {
        database = FirebaseDatabase.getInstance();
        myRef = database.getReference("parking");

        mValueEventListener = new ValueEventListener() {
            @Override
            public void onDataChange(DataSnapshot dataSnapshot) {
                DataSnapshot childSnapshot = dataSnapshot.child(_parkingID);

                String name = childSnapshot.child("name").getValue(String.class);
                Log.d("name", name);
                Float latitude = childSnapshot.child("latitude").getValue(Float.class);
                Log.d("lat",  String.valueOf(latitude));
                Float longitude = childSnapshot.child("longitude").getValue(Float.class);
                Log.d("long", String.valueOf(longitude));
                TYPE type = TYPE.valueOf(childSnapshot.child("type").getValue(String.class));
                TYPE_VEHICLE typeVehicle = TYPE_VEHICLE.valueOf(childSnapshot.child("typeVehicle").getValue(String.class));
                TYPE_STRUCTURE typeStructure = TYPE_STRUCTURE.valueOf(childSnapshot.child("typeStructure").getValue(String.class));
                Integer nSecFloor = childSnapshot.child("nSecFloor").getValue(Integer.class);
                Log.d("nSecFloor", String.valueOf(nSecFloor));
                List<TYPE_LINE> appLine = new ArrayList<>();
                for (DataSnapshot lineSnapshot : childSnapshot.child("typeLine").getChildren())
                {
                    String a = lineSnapshot.getKey();
                    appLine.add(TYPE_LINE.valueOf(a));
                }
                /*Read issue*/

                List<Parking_slot> slot_app = new ArrayList<>();
                for (DataSnapshot slotSnapshot : childSnapshot.child("slot").getChildren())
                {
                    String a = slotSnapshot.getKey();
                    String value = slotSnapshot.child("value").getValue(String.class);
                    TYPE_LINE type_s = TYPE_LINE.valueOf(slotSnapshot.child("type").getValue(String.class));
                    Integer status = slotSnapshot.child("status").getValue(Integer.class);
                    Parking_slot app = new Parking_slot(a, value, type_s, status);
                    slot_app.add(app);
                }

                Comparator<Parking_slot> customComparator = (o1, o2) -> {
                    String s1 = o1.id();
                    String s2 = o2.id();

                    // Extract the numeric part and compare as integers
                    int num1 = Integer.parseInt(s1.substring(1));
                    int num2 = Integer.parseInt(s2.substring(1));

                    return Integer.compare(num1, num2);
                };
                Collections.sort(slot_app, customComparator);
                for (int i = 0; i < slot_app.size(); i++)
                {
                    Log.d("SLOT", String.valueOf(slot_app.get(i).id()));
                }

                Float costH = childSnapshot.child("costH").getValue(Float.class);

                Float end_latitude = childSnapshot.child("Endlatitude").getValue(Float.class);
                Float end_longitude = childSnapshot.child("Endlongitude").getValue(Float.class);
                String street_name = childSnapshot.child("streetName").getValue(String.class);

                parking = new Parking(_parkingID, name, typeStructure, nSecFloor, slot_app, new Pair<>(latitude, longitude), type, typeVehicle, appLine, costH, new Pair<>(end_latitude, end_longitude), street_name);
                Log.d("costH", String.valueOf(parking.costHourly()));

                TextView parkingName = findViewById(R.id.name);
                parkingName.setText(parking.name());
                txt_street.setText(parking.streetName());
                num_slot = findViewById(R.id.num_slot);
                num_slot.setText(String.valueOf(parking.slot().size()));
                txt_prize = findViewById(R.id.txt_prize);
                txt_perhour = findViewById(R.id.asads);
                if (parking.costHourly().equals(0F))
                {
                    txt_prize.setText("Free");
                    txt_perhour.setText("parking");
                }
                else {
                    txt_prize.setText("€" + String.valueOf(parking.costHourly()));
                    txt_perhour.setText("per hour");
                }


                if (parking.typeVehicle().equals(TYPE_VEHICLE.car))
                {
                    //img_motor.setAlpha(0);
                    img_motor.setImageResource(R.drawable.icon_motor_grey);
                }
                else if(parking.typeVehicle().equals(TYPE_VEHICLE.motorcycle))
                {
                    //img_car.setAlpha(0);
                    img_car.setImageResource(R.drawable.icon_car_grey);
                }
                else {
                    img_motor.setImageResource(R.drawable.icon_motor_blue);
                    img_car.setImageResource(R.drawable.icon_car_blue);
                }

                boolean y = false;
                boolean g = false;
                boolean p = false;
                for (int i = 0; i < parking.typeLine().size(); i++)
                {
                    if (parking.typeLine().get(i).equals(TYPE_LINE.yellow))
                    {
                        y = true;
                    }
                    if (parking.typeLine().get(i).equals(TYPE_LINE.green))
                    {
                        g = true;
                    }
                    if (parking.typeLine().get(i).equals(TYPE_LINE.pink))
                    {
                        p = true;
                    }

                }

                int iconResource1 = y ? R.drawable.border_disabled_selected : R.drawable.border_disabled_grey;
                img_yellow.setImageResource(iconResource1);
                int iconResource2 = g ? R.drawable.border_charging_selected : R.drawable.border_charging_grey;
                img_charge.setImageResource(iconResource2);
                int iconResource3 = p ? R.drawable.border_pink_selected : R.drawable.border_pink_grey;
                img_pink.setImageResource(iconResource3);
                /*if (!y)
                {
                    img_yellow.setAlpha(0);
                }
                if (!g)
                {
                    img_charge.setAlpha(0);
                }
                if (!p)
                {
                    img_pink.setAlpha(0);
                }*/

                Integer notAvSlot = 0;
                for (int i = 0; i < parking.slot().size(); i++)
                {
                    if (parking.slot().get(i).status() == 1)
                    {
                        notAvSlot++;
                    }
                }
                //Log.d("onCancelled", String.valueOf(parking.slot().size()) + "; " + String.valueOf(notAvSlot));

                if (User.ongoing().size() > 0)
                {
                    if (User.ongoing().get(0).ongoing().equals("true"))
                    {
                        if (parking.slot().size() - notAvSlot > 0) //there are available slot
                        {
                            available_slot.setTextColor(getResources().getColor(R.color.green));
                            available_slot.setText("Slot Available, parking occupation: " + String.valueOf(notAvSlot) + "/" + String.valueOf(parking.slot().size()));

                        } else {
                            available_slot.setTextColor(getResources().getColor(R.color.red));
                            available_slot.setText("Parking full");
                        }
                        btn_pay.setText("you can't pay, because you already have an ongoing parking");
                        btn_pay.setEnabled(false);
                        btn_pay.setBackgroundResource(R.drawable.btn_background_3);
                    }
                    else {
                        if (parking.slot().size() - notAvSlot > 0) //there are available slot
                        {
                            btn_pay.setEnabled(true);
                            btn_pay.setBackgroundResource(R.drawable.btn_background_2);
                            available_slot.setTextColor(getResources().getColor(R.color.green));
                            available_slot.setText("Slot Available, parking occupation: " + String.valueOf(notAvSlot) + "/" + String.valueOf(parking.slot().size()));

                            if (parking.costHourly().equals(0F)) {
                                btn_pay.setEnabled(false);
                                btn_pay.setBackgroundResource(R.drawable.btn_background_3);
                            } else {
                                btn_pay.setEnabled(true);
                                btn_pay.setBackgroundResource(R.drawable.btn_background_2);
                            }

                        } else {
                            btn_pay.setEnabled(false);
                            btn_pay.setBackgroundResource(R.drawable.btn_background_3);
                            available_slot.setTextColor(getResources().getColor(R.color.red));
                            available_slot.setText("Parking full");

                        }
                    }
                }
                else {
                    if (parking.slot().size() - notAvSlot > 0) //there are available slot
                    {
                        btn_pay.setEnabled(true);
                        btn_pay.setBackgroundResource(R.drawable.btn_background_2);
                        available_slot.setTextColor(getResources().getColor(R.color.green));
                        available_slot.setText("Slot Available, parking occupation: " + String.valueOf(notAvSlot) + "/" + String.valueOf(parking.slot().size()));

                        if (parking.costHourly().equals(0F)) {
                            btn_pay.setEnabled(false);
                            btn_pay.setBackgroundResource(R.drawable.btn_background_3);
                        } else {
                            btn_pay.setEnabled(true);
                            btn_pay.setBackgroundResource(R.drawable.btn_background_2);
                        }

                    } else {
                        btn_pay.setEnabled(false);
                        btn_pay.setBackgroundResource(R.drawable.btn_background_3);
                        available_slot.setTextColor(getResources().getColor(R.color.red));
                        available_slot.setText("Parking full");

                    }
                }



            }

            @Override
            public void onCancelled(DatabaseError databaseError) {
                Log.e(TAG, "onCancelled", databaseError.toException());
            }
        };

        myRef.addValueEventListener(mValueEventListener);

    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        myRef.removeEventListener(mValueEventListener);
    }
    //disable back button
    @Override
    public void onBackPressed(){}

}
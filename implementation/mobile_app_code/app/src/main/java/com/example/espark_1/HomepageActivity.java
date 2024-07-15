package com.example.espark_1;

import static android.content.ContentValues.TAG;

import android.content.DialogInterface;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.graphics.drawable.BitmapDrawable;
import android.graphics.drawable.ColorDrawable;
import android.graphics.drawable.Drawable;
import android.location.LocationListener;
import android.location.LocationManager;
import android.os.AsyncTask;
import android.os.Bundle;
import android.provider.Settings;
import android.util.Log;

import com.example.espark_1.databinding.ActivityHomepageBinding;
import com.google.firebase.database.DataSnapshot;
import com.google.firebase.database.DatabaseError;
import com.google.firebase.database.DatabaseReference;
import com.google.firebase.database.FirebaseDatabase;
import com.google.firebase.database.ValueEventListener;

import android.Manifest;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.preference.PreferenceManager;
import android.util.Pair;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.TextView;
import android.widget.Toast;

import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.widget.AppCompatButton;
import androidx.appcompat.widget.AppCompatImageView;
import androidx.appcompat.widget.AppCompatToggleButton;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;
import org.osmdroid.api.IMapController;
import org.osmdroid.config.Configuration;
import org.osmdroid.tileprovider.tilesource.TileSourceFactory;
import org.osmdroid.util.GeoPoint;
import org.osmdroid.views.MapView;
import org.osmdroid.views.overlay.Marker;
import org.osmdroid.views.overlay.Polyline;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.Serializable;
import java.net.HttpURLConnection;
import java.net.URL;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.Date;
import java.util.List;
import java.util.Objects;
import java.util.Vector;


public class HomepageActivity extends DrawerBaseActivity {


    private ActivityHomepageBinding binding;
    private final int REQUEST_PERMISSIONS_REQUEST_CODE = 1;
    private MapView map = null;
    private FirebaseDatabase database;
    private DatabaseReference myRef;
    private ValueEventListener mValueEventListener;
    private List<Pair<String, Marker>> markerList;
    private AppCompatButton txt_ongoing;
    private AppCompatImageView _btn_filter;
    private Float max_prize = -1F;
    private boolean filterCar = true;
    private boolean filterMotorcycle = true;
    private boolean filterWhite = false;
    private boolean filterBlue = false;
    private boolean filterPink = false;
    private boolean filterGreen = false;
    private boolean filterYellow = false;

    AppCompatToggleButton toggleButtonCar;
    AppCompatToggleButton toggleButtonMotor;
    AppCompatToggleButton toggleButtonWhite;
    AppCompatToggleButton toggleButtonBlue;
    AppCompatToggleButton toggleButtonGreen;
    AppCompatToggleButton toggleButtonYellow;
    AppCompatToggleButton toggleButtonPink;
    LinearLayout l_page;
    TextView txt_p_name;
    TextView txt_street;
    TextView txt_tot_slot;
    TextView available_slot;
    AppCompatButton btn_parking_page;
    AppCompatImageView btn_cancel;
    boolean filter_active = false;

    AppCompatButton btn_camera;
    private LocationManager locationManager;
    Marker _userMarker;
    Context mContext;

    AppCompatButton btn_cancel_search;
    AppCompatImageView btn_news;

    private static final String NOMINATIM_API_URL = "https://nominatim.openstreetmap.org/search?format=json&q=";


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        Intent intent = getIntent();

        // Retrieve extras from the intent
        if (intent != null) {
            filterCar = intent.getBooleanExtra("car", true);
            filterMotorcycle = intent.getBooleanExtra("motor", true);
            filterPink = intent.getBooleanExtra("pink", false);
            filterGreen = intent.getBooleanExtra("charge", false);
            filterYellow = intent.getBooleanExtra("yellow", false);
            max_prize = intent.getFloatExtra("max_prize", -1F);
            filter_active = intent.getBooleanExtra("filterActive", false);
        }
        //Log.d("HOMEPAGE", String.valueOf(filterYellow));
        //Log.d("HOMEPAGE", String.valueOf(max_prize));
        if (max_prize.equals(-1F))
        {
            filterWhite = false;
            filterBlue = false;
            Log.d("HOMEPAGE", "wewewe");
        }
        else {
            if (max_prize.equals(0F)) {
                filterWhite = true;
                filterBlue = false;
            } else {
                filterWhite = false;
                filterBlue = true;
            }
        }

        markerList = new ArrayList<>();

        Context ctx = getApplicationContext();
        Configuration.getInstance().load(ctx, PreferenceManager.getDefaultSharedPreferences(ctx));

        binding = ActivityHomepageBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());

        map = (MapView) findViewById(R.id.map);
        map.setTileSource(TileSourceFactory.MAPNIK);

        String strArray[] = Manifest.permission.WRITE_EXTERNAL_STORAGE.split(" ");

        IMapController mapController = map.getController();
        mapController.setZoom(11.5);

        GeoPoint startPoint = new GeoPoint(41.902782, 12.496366);
        mapController.setCenter(startPoint);

        l_page = findViewById(R.id.layout_page);
        l_page.setAlpha(0);
        txt_p_name = findViewById(R.id.name);
        txt_street = findViewById(R.id.txt_street);
        txt_tot_slot = findViewById(R.id.num_slot);
        available_slot = findViewById(R.id.txt_slotAvailable);
        btn_parking_page = findViewById(R.id.button_page);
        btn_parking_page.setEnabled(false);
        btn_cancel = findViewById(R.id.btn_cancel);
        btn_cancel.setEnabled(false);
        btn_news = findViewById(R.id.btn_news);

        markerExtractor();

        //map.setBuiltInZoomControls(true);
        map.setMultiTouchControls(true);
        requestPermissionsIfNecessary(strArray);

        txt_ongoing = findViewById(R.id.button_ongoing);
        if (!User.ongoing().isEmpty()) {
            if (User.ongoing().get(0).ongoing().equals("true")) {
                //txt_ongoing.setText("Ongoing parking: " + User.ongoing().get(0).name());
                txt_ongoing.setText("You have active parking session");
                txt_ongoing.setEnabled(true);
                txt_ongoing.setAlpha(1);
            } else if (User.ongoing().get(0).ongoing().equals("false")) {
                txt_ongoing.setText("");
                txt_ongoing.setEnabled(false);
                txt_ongoing.setAlpha(0);
            }
        }
        txt_ongoing.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(HomepageActivity.this, Favorite_parkingActivity.class);
                intent.putExtra("section", "ongoing");
                startActivity(intent);
            }
        });

        _btn_filter = findViewById(R.id.btn_filter);
        _btn_filter.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(HomepageActivity.this, FilterActivity.class);
                intent.putExtra("car", filterCar);
                intent.putExtra("motor", filterMotorcycle);
                intent.putExtra("charge", filterGreen);
                intent.putExtra("pink", filterPink);
                intent.putExtra("yellow", filterYellow);
                intent.putExtra("max_prize", max_prize);
                startActivity(intent);
            }
        });

        map.invalidate();

        LinearLayout l_search = findViewById(R.id.layout_search);
        EditText searchEditText = findViewById(R.id.search);
        btn_cancel_search = findViewById(R.id.btn_cancel_search);
        searchEditText.setEnabled(false);

        AppCompatButton searchEditTextClick = findViewById(R.id.search_click);

        AppCompatButton searchButton = findViewById(R.id.button_tosearch);
        searchButton.setEnabled(false);

        searchEditTextClick.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                l_search.setAlpha(1);
                btn_cancel_search.setAlpha(1);
                btn_cancel_search.setEnabled(true);
                searchEditText.setEnabled(true);
                searchButton.setEnabled(true);
                txt_ongoing.setEnabled(false);
                _btn_filter.setEnabled(false);
                btn_news.setEnabled(false);
                searchEditTextClick.setEnabled(false);
                btn_camera.setEnabled(false);
            }
        });
        // Set an OnClickListener for the search button
        searchButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                String searchText = searchEditText.getText().toString().trim();

                if (!searchText.isEmpty()) {
                    new GeocodeTask().execute(searchText);
                }
                l_search.setAlpha(0);
                btn_cancel_search.setAlpha(0);
                btn_cancel_search.setEnabled(false);
                searchEditText.setEnabled(false);
                searchButton.setEnabled(false);
                txt_ongoing.setEnabled(true);
                _btn_filter.setEnabled(true);
                btn_news.setEnabled(true);
                searchEditTextClick.setEnabled(true);
                btn_camera.setEnabled(true);
            }
        });

        btn_cancel_search.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                l_search.setAlpha(0);
                btn_cancel_search.setAlpha(0);
                btn_cancel_search.setEnabled(false);
                searchEditText.setEnabled(false);
                searchButton.setEnabled(false);
                txt_ongoing.setEnabled(true);
                _btn_filter.setEnabled(true);
                btn_news.setEnabled(true);
                searchEditTextClick.setEnabled(true);
                btn_camera.setEnabled(true);
            }
        });

        markerUpdateFilter();

        btn_news.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                new NewsTask(HomepageActivity.this).execute();
            }
        });

        userChange();

        _userMarker = new Marker(map);

        mContext = this;
        locationManager = (LocationManager) mContext.getSystemService(Context.LOCATION_SERVICE);
        if (ActivityCompat.checkSelfPermission(this, android.Manifest.permission.ACCESS_FINE_LOCATION) != PackageManager.PERMISSION_GRANTED && ActivityCompat.checkSelfPermission(this, android.Manifest.permission.ACCESS_COARSE_LOCATION) != PackageManager.PERMISSION_GRANTED) {
            // TODO: Consider calling
            //    ActivityCompat#requestPermissions
            // here to request the missing permissions, and then overriding
            //   public void onRequestPermissionsResult(int requestCode, String[] permissions,
            //                                          int[] grantResults)
            // to handle the case where the user grants the permission. See the documentation
            // for ActivityCompat#requestPermissions for more details.
            return;
        }
        locationManager.requestLocationUpdates(LocationManager.NETWORK_PROVIDER,
                2000,
                10, locationListenerGPS);
        isLocationEnabled();

        btn_camera = findViewById(R.id.button_qrcode);
        btn_camera.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent in = new Intent(HomepageActivity.this, QRcodeActivity.class);
                startActivity(in);
            }
        });

    }
    //disable back button
    @Override
    public void onBackPressed(){}
    @Override
    public void onResume() {
        super.onResume();
        //this will refresh the osmdroid configuration on resuming.
        //if you make changes to the configuration, use
        //SharedPreferences prefs = PreferenceManager.getDefaultSharedPreferences(this);
        //Configuration.getInstance().load(this, PreferenceManager.getDefaultSharedPreferences(this));
        map.onResume(); //needed for compass, my location overlays, v6.0.0 and up

        isLocationEnabled();
    }

    @Override
    public void onPause() {
        super.onPause();
        //this will refresh the osmdroid configuration on resuming.
        //if you make changes to the configuration, use
        //SharedPreferences prefs = PreferenceManager.getDefaultSharedPreferences(this);
        //Configuration.getInstance().save(this, prefs);
        map.onPause();  //needed for compass, my location overlays, v6.0.0 and up
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        ArrayList<String> permissionsToRequest = new ArrayList<>();
        for (int i = 0; i < grantResults.length; i++) {
            permissionsToRequest.add(permissions[i]);
        }
        if (permissionsToRequest.size() > 0) {
            ActivityCompat.requestPermissions(
                    this,
                    permissionsToRequest.toArray(new String[0]),
                    REQUEST_PERMISSIONS_REQUEST_CODE);
        }
    }

    private void requestPermissionsIfNecessary(String[] permissions) {
        ArrayList<String> permissionsToRequest = new ArrayList<>();
        for (String permission : permissions) {
            if (ContextCompat.checkSelfPermission(this, permission)
                    != PackageManager.PERMISSION_GRANTED) {
                // Permission is not granted
                permissionsToRequest.add(permission);
            }
        }
        if (permissionsToRequest.size() > 0) {
            ActivityCompat.requestPermissions(
                    this,
                    permissionsToRequest.toArray(new String[0]),
                    REQUEST_PERMISSIONS_REQUEST_CODE);
        }
    }

    LocationListener locationListenerGPS=new LocationListener() {
        @Override
        public void onLocationChanged(android.location.Location location) {
            double latitude=location.getLatitude();
            double longitude=location.getLongitude();
            String msg="New Latitude: "+latitude + "New Longitude: "+longitude;
            Log.d("WEWEWEWEWEWE", msg);
            //Toast.makeText(mContext,msg,Toast.LENGTH_LONG).show();

            Context context = map.getContext();
            Bitmap iconBitmap = BitmapFactory.decodeResource(context.getResources(),
                    R.drawable.gpsicon);
            BitmapDrawable iconDrawable = new BitmapDrawable(context.getResources(),
                    iconBitmap);
            GeoPoint point = new GeoPoint(latitude, longitude);
            _userMarker.setPosition(point);
            _userMarker.setAnchor(Marker.ANCHOR_CENTER, Marker.ANCHOR_BOTTOM);
            _userMarker.setIcon(iconDrawable);

            map.getOverlays().add(_userMarker);
        }

        @Override
        public void onStatusChanged(String provider, int status, Bundle extras) {

        }

        @Override
        public void onProviderEnabled(String provider) {

        }

        @Override
        public void onProviderDisabled(String provider) {

        }
    };

    private void isLocationEnabled() {

        if(!locationManager.isProviderEnabled(LocationManager.GPS_PROVIDER)){
            AlertDialog.Builder alertDialog=new AlertDialog.Builder(mContext);
            alertDialog.setTitle("Enable Location");
            alertDialog.setMessage("Your locations setting is not enabled. Please enabled it in settings menu.");
            alertDialog.setPositiveButton("Location Settings", new DialogInterface.OnClickListener(){
                public void onClick(DialogInterface dialog, int which){
                    Intent intent=new Intent(Settings.ACTION_LOCATION_SOURCE_SETTINGS);
                    startActivity(intent);
                }
            });
            alertDialog.setNegativeButton("Cancel", new DialogInterface.OnClickListener(){
                public void onClick(DialogInterface dialog, int which){
                    dialog.cancel();
                }
            });
            AlertDialog alert=alertDialog.create();
            alert.show();
        }
        else{
            /*AlertDialog.Builder alertDialog=new AlertDialog.Builder(mContext);
            alertDialog.setTitle("Confirm Location");
            alertDialog.setMessage("Your Location is enabled, please enjoy");
            alertDialog.setNegativeButton("Back to interface",new DialogInterface.OnClickListener(){
                public void onClick(DialogInterface dialog, int which){
                    dialog.cancel();
                }
            });
            AlertDialog alert=alertDialog.create();
            alert.show();*/
        }
    }


    public void markerExtractor()
    {
        database = FirebaseDatabase.getInstance();
        myRef = database.getReference("parking");
        
        //_markerList = new ArrayList<>();
        
        myRef.addListenerForSingleValueEvent(new ValueEventListener() {
            @Override
            public void onDataChange(DataSnapshot dataSnapshot) {
                for (DataSnapshot childSnapshot : dataSnapshot.getChildren())
                {
                    String parkingId = childSnapshot.getKey();
                    Float parkinglatitude = childSnapshot.child("latitude").getValue(Float.class);
                    Float parkinglongitude = childSnapshot.child("longitude").getValue(Float.class);
                    markerCreate(parkingId, parkinglatitude, parkinglongitude);
                }
            }

            @Override
            public void onCancelled(DatabaseError databaseError) {
                Log.e(TAG, "onCancelled", databaseError.toException());
            }
        });

    }

    public void markerCreate(String id, Float latitude, Float longitude)
    {
        // Get a reference to the current context
        Context context = map.getContext();

        // Load the icon bitmap from a resource file
        Bitmap iconBitmap = BitmapFactory.decodeResource(context.getResources(),
                R.drawable.marker_icon);
        // Calculate the desired size in pixels
        int desiredSize = 80; // for example

        // Calculate the scale factor for the icon
        float scale = (float) desiredSize / iconBitmap.getHeight();

        // Create the scaled icon bitmap
        Bitmap scaledBitmap = Bitmap.createScaledBitmap(iconBitmap,
                (int) (iconBitmap.getWidth() * scale),
                (int) (iconBitmap.getHeight() * scale),
                true);

        // Create a new bitmap drawable with the icon bitmap
        BitmapDrawable iconDrawable = new BitmapDrawable(context.getResources(),
                scaledBitmap);

        Marker marker = new Marker(map);
        GeoPoint point = new GeoPoint(latitude, longitude);
        Log.d("VALUE", String.valueOf(id));
        marker.setPosition(point);
        marker.setAnchor(Marker.ANCHOR_CENTER, Marker.ANCHOR_BOTTOM);

        marker.setIcon(iconDrawable);

        marker.setOnMarkerClickListener(new Marker.OnMarkerClickListener() {
            @Override
            public boolean onMarkerClick(Marker marker, MapView mapView) {

                Context context1 = map.getContext();
                Bitmap iconBitmap1 = BitmapFactory.decodeResource(context1.getResources(),
                        R.drawable.marker_icon_highlighted);
                // Calculate the desired size in pixels
                int desiredSize = 130; // for example

                // Calculate the scale factor for the icon
                float scale = (float) desiredSize / iconBitmap1.getHeight();

                // Create the scaled icon bitmap
                Bitmap scaledBitmap1 = Bitmap.createScaledBitmap(iconBitmap1,
                        (int) (iconBitmap1.getWidth() * scale),
                        (int) (iconBitmap1.getHeight() * scale),
                        true);

                // Create a new bitmap drawable with the icon bitmap
                BitmapDrawable iconDrawable1 = new BitmapDrawable(context1.getResources(),
                        scaledBitmap1);
                marker.setIcon(iconDrawable1);
                for (int i = 0; i < markerList.size(); i++)
                {
                    if (!markerList.get(i).first.equals(id)) {
                        Bitmap iconBitmap2 = BitmapFactory.decodeResource(context1.getResources(),
                                R.drawable.marker_icon);
                        // Calculate the desired size in pixels
                        int desiredSize1 = 80; // for example

                        // Calculate the scale factor for the icon
                        float scale1 = (float) desiredSize1 / iconBitmap2.getHeight();

                        // Create the scaled icon bitmap
                        Bitmap scaledBitmap2 = Bitmap.createScaledBitmap(iconBitmap2,
                                (int) (iconBitmap2.getWidth() * scale1),
                                (int) (iconBitmap2.getHeight() * scale1),
                                true);

                        // Create a new bitmap drawable with the icon bitmap
                        BitmapDrawable iconDrawable2 = new BitmapDrawable(context1.getResources(),
                                scaledBitmap2);
                        markerList.get(i).second.setIcon(iconDrawable2);
                    }
                }

                short_page(id);
                return false;
            }

        });
        markerList.add(new Pair<>(id, marker));
        map.getOverlays().add(marker);

    }

    public void short_page(String parking_id)
    {
        l_page.setAlpha(1);
        btn_camera.setEnabled(false);
        btn_camera.setAlpha(0);

        btn_parking_page.setEnabled(true);
        btn_parking_page.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(HomepageActivity.this, ParkingActivity.class);
                intent.putExtra("parking", parking_id);
                startActivity(intent);
            }
        });

        btn_cancel.setEnabled(true);
        btn_cancel.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Context context1 = map.getContext();
                for (int i = 0; i < markerList.size(); i++)
                {

                    Bitmap iconBitmap1 = BitmapFactory.decodeResource(context1.getResources(),
                            R.drawable.marker_icon);
                    // Calculate the desired size in pixels
                    int desiredSize = 80; // for example

                    // Calculate the scale factor for the icon
                    float scale = (float) desiredSize / iconBitmap1.getHeight();

                    // Create the scaled icon bitmap
                    Bitmap scaledBitmap1 = Bitmap.createScaledBitmap(iconBitmap1,
                            (int) (iconBitmap1.getWidth() * scale),
                            (int) (iconBitmap1.getHeight() * scale),
                            true);

                    // Create a new bitmap drawable with the icon bitmap
                    BitmapDrawable iconDrawable1 = new BitmapDrawable(context1.getResources(),
                            scaledBitmap1);
                    markerList.get(i).second.setIcon(iconDrawable1);
                }
                l_page.setAlpha(0);
                btn_parking_page.setEnabled(false);
                btn_cancel.setEnabled(false);
                btn_camera.setEnabled(true);
                btn_camera.setAlpha(1);
            }
        });
        database = FirebaseDatabase.getInstance();
        myRef = database.getReference("parking");

        myRef.addListenerForSingleValueEvent(new ValueEventListener() {
            @Override
            public void onDataChange(DataSnapshot dataSnapshot) {
                DataSnapshot childSnapshot = dataSnapshot.child(parking_id);

                String name = childSnapshot.child("name").getValue(String.class);
                Log.d("name", name);

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

                String street_name = childSnapshot.child("streetName").getValue(String.class);

                txt_p_name.setText(name);
                txt_street.setText(street_name);
                txt_tot_slot.setText(String.valueOf(slot_app.size()));

                Integer notAvSlot = 0;
                for (int i = 0; i < slot_app.size(); i++)
                {
                    if (slot_app.get(i).status() == 1)
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
        });

    }
    public void userChange()
    {
        database = FirebaseDatabase.getInstance();
        myRef = database.getReference("user");

        mValueEventListener = new ValueEventListener() {
            @Override
            public void onDataChange(DataSnapshot dataSnapshot) {
                for (DataSnapshot childSnapshot : dataSnapshot.getChildren())
                {
                    String id = childSnapshot.getKey();
                    String userUID = childSnapshot.child("uid").getValue(String.class);
                    if (!User.ongoing().isEmpty()) {
                        if (id.equals(User.id())) {
                            for (DataSnapshot lineSnapshot : childSnapshot.child("ongoingParking").getChildren()) {
                                User.ongoing().get(0).set_ongoing(lineSnapshot.child("ongoing").getValue(String.class));
                            }
                        }
                    }
                }

                if (!User.ongoing().isEmpty()) {
                    if (User.ongoing().get(0).ongoing().equals("true")) {
                        //txt_ongoing.setText("Ongoing parking: " + User.ongoing().get(0).name());
                        txt_ongoing.setText("You have active parking session");
                        txt_ongoing.setEnabled(true);
                        txt_ongoing.setAlpha(1);
                    } else if (User.ongoing().get(0).ongoing().equals("false")) {
                        txt_ongoing.setText("");
                        txt_ongoing.setEnabled(false);
                        txt_ongoing.setAlpha(0);
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

    private void markerUpdateFilter()
    {
        if (!filter_active)
        {
            Context context = map.getContext();
            for (Pair<String, Marker> pair : markerList)
            {
                // Load the icon bitmap from a resource file
                Bitmap iconBitmap = BitmapFactory.decodeResource(context.getResources(),
                        R.drawable.marker_icon);
                // Calculate the desired size in pixels
                int desiredSize = 80; // for example

                // Calculate the scale factor for the icon
                float scale = (float) desiredSize / iconBitmap.getHeight();

                // Create the scaled icon bitmap
                Bitmap scaledBitmap = Bitmap.createScaledBitmap(iconBitmap,
                        (int) (iconBitmap.getWidth() * scale),
                        (int) (iconBitmap.getHeight() * scale),
                        true);

                // Create a new bitmap drawable with the icon bitmap
                BitmapDrawable iconDrawable = new BitmapDrawable(context.getResources(),
                        scaledBitmap);

                pair.second.setIcon(iconDrawable);

            }

            return;
        }
        Vector<String> markerFilter = new Vector<>();

        database = FirebaseDatabase.getInstance();
        myRef = database.getReference("parking");

        myRef.addListenerForSingleValueEvent(new ValueEventListener() {
            @Override
            public void onDataChange(DataSnapshot dataSnapshot) {
                for (DataSnapshot childSnapshot : dataSnapshot.getChildren())
                {
                    String parkingId = childSnapshot.getKey();
                    Float p = childSnapshot.child("costH").getValue(Float.class);
                    TYPE_VEHICLE typeVehicle = TYPE_VEHICLE.valueOf(childSnapshot.child("typeVehicle").getValue(String.class));

                    if (filterCar && (typeVehicle == TYPE_VEHICLE.car ||  typeVehicle == TYPE_VEHICLE.all))
                    {
                        List<TYPE_LINE> appLine = new ArrayList<>();
                        for (DataSnapshot lineSnapshot : childSnapshot.child("typeLine").getChildren()) {
                            TYPE_LINE a = TYPE_LINE.valueOf(lineSnapshot.getKey());

                            if (filterWhite && a == TYPE_LINE.white) {
                                for (DataSnapshot lineSnapshot1 : childSnapshot.child("typeLine").getChildren()) {
                                    TYPE_LINE a1 = TYPE_LINE.valueOf(lineSnapshot1.getKey());
                                    if (filterPink && a1 == TYPE_LINE.pink) {
                                        markerFilter.add(parkingId);
                                        break;
                                    } else if (filterYellow && a1 == TYPE_LINE.yellow) {
                                        markerFilter.add(parkingId);
                                        break;
                                    } else if (filterGreen && a1 == TYPE_LINE.green) {
                                        markerFilter.add(parkingId);
                                        break;
                                    } else if (!filterGreen && !filterYellow && !filterPink) {
                                        markerFilter.add(parkingId);
                                        break;
                                    }
                                }
                            } else if (filterBlue && a == TYPE_LINE.blue) {
                                if (p <= max_prize)
                                {
                                    for (DataSnapshot lineSnapshot1 : childSnapshot.child("typeLine").getChildren()) {
                                        TYPE_LINE a1 = TYPE_LINE.valueOf(lineSnapshot1.getKey());
                                        if (filterPink && a1 == TYPE_LINE.pink) {
                                            markerFilter.add(parkingId);
                                            break;
                                        } else if (filterYellow && a1 == TYPE_LINE.yellow) {
                                            markerFilter.add(parkingId);
                                            break;
                                        } else if (filterGreen && a1 == TYPE_LINE.green) {
                                            markerFilter.add(parkingId);
                                            break;
                                        } else if (!filterGreen && !filterYellow && !filterPink) {
                                            markerFilter.add(parkingId);
                                            break;
                                        }
                                    }
                                }
                                break;
                            }
                        }
                    }
                    else if (filterMotorcycle && (typeVehicle == TYPE_VEHICLE.motorcycle ||  typeVehicle == TYPE_VEHICLE.all))
                    {
                        List<TYPE_LINE> appLine = new ArrayList<>();
                        for (DataSnapshot lineSnapshot : childSnapshot.child("typeLine").getChildren()) {
                            TYPE_LINE a = TYPE_LINE.valueOf(lineSnapshot.getKey());

                            if (filterWhite && a == TYPE_LINE.white) {
                                for (DataSnapshot lineSnapshot1 : childSnapshot.child("typeLine").getChildren()) {
                                    TYPE_LINE a1 = TYPE_LINE.valueOf(lineSnapshot1.getKey());
                                    if (filterPink && a1 == TYPE_LINE.pink) {
                                        markerFilter.add(parkingId);
                                        break;
                                    } else if (filterYellow && a1 == TYPE_LINE.yellow) {
                                        markerFilter.add(parkingId);
                                        break;
                                    } else if (filterGreen && a1 == TYPE_LINE.green) {
                                        markerFilter.add(parkingId);
                                        break;
                                    } else if (!filterGreen && !filterYellow && !filterPink) {
                                        markerFilter.add(parkingId);
                                        break;
                                    }
                                }
                            } else if (filterBlue && a == TYPE_LINE.blue) {
                                if (p <= max_prize)
                                {
                                    for (DataSnapshot lineSnapshot1 : childSnapshot.child("typeLine").getChildren()) {
                                        TYPE_LINE a1 = TYPE_LINE.valueOf(lineSnapshot1.getKey());
                                        if (filterPink && a1 == TYPE_LINE.pink) {
                                            markerFilter.add(parkingId);
                                            break;
                                        } else if (filterYellow && a1 == TYPE_LINE.yellow) {
                                            markerFilter.add(parkingId);
                                            break;
                                        } else if (filterGreen && a1 == TYPE_LINE.green) {
                                            markerFilter.add(parkingId);
                                            break;
                                        } else if (!filterGreen && !filterYellow && !filterPink) {
                                            markerFilter.add(parkingId);
                                            break;
                                        }
                                    }
                                }
                                break;
                            }
                        }
                    }
                    else if (filterMotorcycle && filterCar)
                    {
                        List<TYPE_LINE> appLine = new ArrayList<>();
                        for (DataSnapshot lineSnapshot : childSnapshot.child("typeLine").getChildren()) {
                            TYPE_LINE a = TYPE_LINE.valueOf(lineSnapshot.getKey());

                            if (filterWhite && a == TYPE_LINE.white) {
                                for (DataSnapshot lineSnapshot1 : childSnapshot.child("typeLine").getChildren()) {
                                    TYPE_LINE a1 = TYPE_LINE.valueOf(lineSnapshot1.getKey());
                                    if (filterPink && a1 == TYPE_LINE.pink) {
                                        markerFilter.add(parkingId);
                                        break;
                                    } else if (filterYellow && a1 == TYPE_LINE.yellow) {
                                        markerFilter.add(parkingId);
                                        break;
                                    } else if (filterGreen && a1 == TYPE_LINE.green) {
                                        markerFilter.add(parkingId);
                                        break;
                                    } else if (!filterGreen && !filterYellow && !filterPink) {
                                        markerFilter.add(parkingId);
                                        break;
                                    }
                                }
                            } else if (filterBlue && a == TYPE_LINE.blue) {
                                if (p <= max_prize)
                                {
                                    for (DataSnapshot lineSnapshot1 : childSnapshot.child("typeLine").getChildren()) {
                                        TYPE_LINE a1 = TYPE_LINE.valueOf(lineSnapshot1.getKey());
                                        if (filterPink && a1 == TYPE_LINE.pink) {
                                            markerFilter.add(parkingId);
                                            break;
                                        } else if (filterYellow && a1 == TYPE_LINE.yellow) {
                                            markerFilter.add(parkingId);
                                            break;
                                        } else if (filterGreen && a1 == TYPE_LINE.green) {
                                            markerFilter.add(parkingId);
                                            break;
                                        } else if (!filterGreen && !filterYellow && !filterPink) {
                                            markerFilter.add(parkingId);
                                            break;
                                        }
                                    }
                                }
                                break;
                            }
                        }
                    }

                }

                Log.d("MARKER FILTER", String.valueOf(markerFilter.size()));


                // Get a reference to the current context
                Context context = map.getContext();
                int i = 0;
                for (Pair<String, Marker> pair : markerList)
                {
                    if (i < markerFilter.size())
                    {
                        if (Objects.equals(pair.first, markerFilter.get(i)))
                        {
                            // Load the icon bitmap from a resource file
                            Bitmap iconBitmap = BitmapFactory.decodeResource(context.getResources(),
                                    R.drawable.marker_icon);
                            // Calculate the desired size in pixels
                            int desiredSize = 80; // for example

                            // Calculate the scale factor for the icon
                            float scale = (float) desiredSize / iconBitmap.getHeight();

                            // Create the scaled icon bitmap
                            Bitmap scaledBitmap = Bitmap.createScaledBitmap(iconBitmap,
                                    (int) (iconBitmap.getWidth() * scale),
                                    (int) (iconBitmap.getHeight() * scale),
                                    true);

                            // Create a new bitmap drawable with the icon bitmap
                            BitmapDrawable iconDrawable = new BitmapDrawable(context.getResources(),
                                    scaledBitmap);

                            pair.second.setIcon(iconDrawable);
                            i++;
                        }
                        else
                        {
                            pair.second.setIcon(getResources().getDrawable(android.R.color.transparent));
                        }
                    }
                    else
                    {
                        // Load the icon bitmap from a resource file
                        Bitmap iconBitmap = BitmapFactory.decodeResource(context.getResources(),
                                R.drawable.marker_icon);
                        // Calculate the desired size in pixels
                        int desiredSize = 80; // for example

                        // Calculate the scale factor for the icon
                        float scale = (float) desiredSize / iconBitmap.getHeight();

                        // Create the scaled icon bitmap
                        Bitmap scaledBitmap = Bitmap.createScaledBitmap(iconBitmap,
                                (int) (iconBitmap.getWidth() * scale),
                                (int) (iconBitmap.getHeight() * scale),
                                true);

                        // Create a new bitmap drawable with the icon bitmap
                        BitmapDrawable iconDrawable = new BitmapDrawable(context.getResources(),
                                scaledBitmap);

                        pair.second.setIcon(getResources().getDrawable(android.R.color.transparent));
                    }

                }


            }

            @Override
            public void onCancelled(DatabaseError databaseError) {
                Log.e(TAG, "onCancelled", databaseError.toException());
            }
        });



    }


    private class GeocodeTask extends AsyncTask<String, Void, String> {
        @Override
        protected String doInBackground(String... params) {
            try {
                String searchText = params[0].replace(" ", "+");
                URL url = new URL(NOMINATIM_API_URL + searchText);
                HttpURLConnection connection = (HttpURLConnection) url.openConnection();
                connection.setRequestMethod("GET");

                BufferedReader reader = new BufferedReader(new InputStreamReader(connection.getInputStream()));
                StringBuilder response = new StringBuilder();
                String line;
                while ((line = reader.readLine()) != null) {
                    response.append(line);
                }
                reader.close();
                return response.toString();
            } catch (IOException e) {
                e.printStackTrace();
            }
            return null;
        }

        @Override
        protected void onPostExecute(String result) {
            if (result != null) {
                try {
                    JSONArray jsonArray = new JSONArray(result);
                    if (jsonArray.length() > 0) {
                        JSONObject firstResult = jsonArray.getJSONObject(0);
                        double lat = firstResult.getDouble("lat");
                        double lon = firstResult.getDouble("lon");

                        // Center the map on the found coordinates
                        GeoPoint streetLocation = new GeoPoint(lat, lon);
                        map.getController().setCenter(streetLocation);
                        IMapController mapController = map.getController();
                        mapController.setZoom(20);
                    } else {
                        Toast.makeText(getApplicationContext(), "Street not found", Toast.LENGTH_SHORT).show();
                    }
                } catch (JSONException e) {
                    e.printStackTrace();
                }
            } else {
                Toast.makeText(getApplicationContext(), "Error while searching", Toast.LENGTH_SHORT).show();
            }
        }
    }

   /* @Override
    protected void onDestroy() {
        super.onDestroy();

    }*/
}
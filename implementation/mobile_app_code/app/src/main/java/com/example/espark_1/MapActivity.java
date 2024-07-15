package com.example.espark_1;

import android.Manifest;
import android.content.Context;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.preference.PreferenceManager;

import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import com.example.espark_1.databinding.ActivityMapBinding;

import org.osmdroid.api.IMapController;
import org.osmdroid.config.Configuration;
import org.osmdroid.tileprovider.tilesource.TileSourceFactory;
import org.osmdroid.util.GeoPoint;
import org.osmdroid.views.MapView;

import java.util.ArrayList;
import java.util.List;

public class MapActivity extends DrawerBaseActivity {

    private ActivityMapBinding binding;

    private final int REQUEST_PERMISSIONS_REQUEST_CODE = 1;
    private MapView map = null;

    List<Parking> parking;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        Context ctx = getApplicationContext();
        Configuration.getInstance().load(ctx, PreferenceManager.getDefaultSharedPreferences(ctx));

        binding = ActivityMapBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());

        map = (MapView) findViewById(R.id.map);
        map.setTileSource(TileSourceFactory.MAPNIK);

        String strArray[] = Manifest.permission.WRITE_EXTERNAL_STORAGE.split(" ");

        IMapController mapController = map.getController();
        mapController.setZoom(11.5);

        GeoPoint startPoint = new GeoPoint(41.902782, 12.496366);
        mapController.setCenter(startPoint);

        // allocate parking
        /*Pair<Float, Float> c1 = new Pair<>(41.9090587f, 12.5303896f);
        List<LINE_TYPE> l1 = new ArrayList<LINE_TYPE>();
        l1.add(LINE_TYPE.WHITE);
        l1.add(LINE_TYPE.BLUE);
        l1.add(LINE_TYPE.PINK);
        Parking p1 = new Parking("P1", true, 3, c1, TYPE.ALL, l1);

        Pair<Float, Float> c2 = new Pair<>(41.8941519f, 12.4932908f);
        List<LINE_TYPE> l2 = new ArrayList<LINE_TYPE>();
        l2.add(LINE_TYPE.WHITE);
        Parking p2 = new Parking("P1", true, 1, c2, TYPE.MOTORCYCLE, l1);

        Pair<Float, Float> c3 = new Pair<>(41.9004825f, 12.4997024f);
        List<LINE_TYPE> l3 = new ArrayList<LINE_TYPE>();
        l3.add(LINE_TYPE.GREEN);
        l3.add(LINE_TYPE.BLUE);
        Parking p3 = new Parking("P1", false, 4, c3, TYPE.CAR, l3);

        parking = new ArrayList<Parking>();
        parking.add(p1);
        parking.add(p2);
        parking.add(p3);

        // add parking in map
        for(int j = 0; j < 3; j++) {

            Marker marker = new Marker(map);
            GeoPoint point = new GeoPoint(parking.get(j).coordinates.first.doubleValue(), parking.get(j).coordinates.second.doubleValue());
            marker.setPosition(point);
            marker.setAnchor(Marker.ANCHOR_CENTER, Marker.ANCHOR_BOTTOM);


            marker.setOnMarkerClickListener(new Marker.OnMarkerClickListener() {
                @Override
                public boolean onMarkerClick(Marker marker, MapView mapView) {
                    startActivity(new Intent(MapActivity.this, ParkingActivity.class));
                    return false;
                }
            });

            map.getOverlays().add(marker);
        }*/

        //map.setBuiltInZoomControls(true);
        map.setMultiTouchControls(true);
        requestPermissionsIfNecessary(strArray);
    }

    @Override
    public void onResume() {
        super.onResume();
        //this will refresh the osmdroid configuration on resuming.
        //if you make changes to the configuration, use
        //SharedPreferences prefs = PreferenceManager.getDefaultSharedPreferences(this);
        //Configuration.getInstance().load(this, PreferenceManager.getDefaultSharedPreferences(this));
        map.onResume(); //needed for compass, my location overlays, v6.0.0 and up
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

    //disable back button
    @Override
    public void onBackPressed(){}
}
package com.example.espark_1;

import android.content.Intent;
import android.view.MenuItem;
import android.view.View;
import android.widget.FrameLayout;
import android.widget.ImageView;

import androidx.annotation.NonNull;
import androidx.appcompat.app.ActionBarDrawerToggle;
import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.Toolbar;
import androidx.core.view.GravityCompat;
import androidx.drawerlayout.widget.DrawerLayout;

import com.google.android.material.navigation.NavigationView;

import java.util.List;

public class DrawerBaseActivityFilter extends AppCompatActivity implements NavigationView.OnNavigationItemSelectedListener {

    DrawerLayout drawerLayout;
    List<String> filter_added_vehicle;
    List<String> filter_added_line;

    @Override
    public void setContentView(View view) {
        drawerLayout = (DrawerLayout) getLayoutInflater().inflate(R.layout.activity_drawer_base_filter, null);
        FrameLayout container = drawerLayout.findViewById((R.id.activityContainer));
        container.addView(view);
        super.setContentView(drawerLayout);

        Toolbar toolbar = drawerLayout.findViewById(R.id.toolBar);
        setSupportActionBar(toolbar);

// Set up the left drawer
        NavigationView navigationView = drawerLayout.findViewById(R.id.nav_view);
        navigationView.setNavigationItemSelectedListener(this);

        ActionBarDrawerToggle leftToggle = new ActionBarDrawerToggle(this, drawerLayout, toolbar, R.string.menu_drawer_open, R.string.menu_drawer_close);
        drawerLayout.addDrawerListener(leftToggle);
        leftToggle.syncState();

        // Duplicate the navigation button for the right drawer
        ImageView rightNavigationButton = new ImageView(this);
        rightNavigationButton.setImageResource(R.drawable.ic_filter); // Replace with your icon resource

        rightNavigationButton.setOnClickListener(v -> {
            if (drawerLayout.isDrawerOpen(GravityCompat.END)) {
                drawerLayout.closeDrawer(GravityCompat.END);
            } else {
                drawerLayout.openDrawer(GravityCompat.END);
            }
        });

        // Add the duplicate navigation button to the right side of the toolbar
        Toolbar.LayoutParams params = new Toolbar.LayoutParams(
                Toolbar.LayoutParams.WRAP_CONTENT,
                Toolbar.LayoutParams.WRAP_CONTENT
        );
        params.gravity = GravityCompat.END;
        params.rightMargin = getResources().getDimensionPixelSize(R.dimen.right_navigation_button_margin); // Adjust this margin as needed

        toolbar.addView(rightNavigationButton, params);

        // Set up the right drawer
        NavigationView navigationViewRight = drawerLayout.findViewById(R.id.nav_view_right);
        navigationViewRight.setNavigationItemSelectedListener(this);

        ActionBarDrawerToggle rightToggle = new ActionBarDrawerToggle(this, drawerLayout, toolbar, R.string.menu_drawer_open, R.string.menu_drawer_close);
        drawerLayout.addDrawerListener(rightToggle);
        rightToggle.syncState();


    }

    @Override
    public boolean onNavigationItemSelected(@NonNull MenuItem item)
    {
        switch (item.getItemId())
        {
            case R.id.nav_home:
                startActivity(new Intent(this, HomepageActivity.class));
                overridePendingTransition(0, 0);
                break;
            case R.id.nav_star:
                //startActivity(new Intent(this, Favorite_parkingActivity.class));
                startActivity(new Intent(this, Favorite_parkingActivity.class).putExtra("section", "favourite"));
                overridePendingTransition(0, 0);
                break;
            case R.id.nav_account:
                startActivity(new Intent(this, AccountActivity.class));
                overridePendingTransition(0, 0);
                break;
            case R.id.car:
                item.setChecked(!item.isChecked());
                filter_added_vehicle.add("car");
                overridePendingTransition(0, 0);
                break;
            case R.id.motor:
                item.setChecked(!item.isChecked());
                filter_added_vehicle.add("motorcycle");
                overridePendingTransition(0, 0);
                break;
            case R.id.nav_white:
                item.setChecked(!item.isChecked());
                filter_added_line.add("white");
                overridePendingTransition(0, 0);
                break;
            case R.id.nav_yellow:
                item.setChecked(!item.isChecked());
                filter_added_line.add("yellow");
                overridePendingTransition(0, 0);
                break;
            case R.id.nav_blue:
                item.setChecked(!item.isChecked());
                filter_added_line.add("blue");
                overridePendingTransition(0, 0);
                break;
            case R.id.nav_pink:
                item.setChecked(!item.isChecked());
                filter_added_line.add("pink");
                overridePendingTransition(0, 0);
                break;
            case R.id.nav_green:
                item.setChecked(!item.isChecked());
                filter_added_line.add("green");
                overridePendingTransition(0, 0);
                break;
            case R.id.button:
                drawerLayout.closeDrawers();
                overridePendingTransition(0, 0);
                break;
        }
        return false;
    }

    protected void allocateActivityTitle(String titleString)
    {
        if (getSupportActionBar() != null)
        {
            getSupportActionBar().setTitle(titleString);
        }
    }
}
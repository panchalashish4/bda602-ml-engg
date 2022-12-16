#!/bin/bash

sleep 30

if mariadb -u root -ppassword123 -h mariadb_ap -e "USE baseball;"
then
    echo "Database exist, creating feature table.."
    mariadb -u root -ppassword123 -h mariadb_ap baseball < features_table.sql
    echo "Feature table successfully created."
else
    echo "Database does not exist, loading database.."
    mariadb -u root -ppassword123 -h mariadb_ap -e "CREATE DATABASE baseball;"
    mariadb -u root -ppassword123 -h mariadb_ap baseball < baseball.sql
    echo "Database Loaded, creating feature table.."
    mariadb -u root -ppassword123 -h mariadb_ap baseball < features_table.sql
    echo "Feature table successfully created."
fi

python3 main.py

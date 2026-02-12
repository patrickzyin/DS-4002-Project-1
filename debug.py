#!/usr/bin/env python3
"""
Detailed inspection of Year column around 2010-2011
"""

import pandas as pd

def main():
    print("Reading SOTU_data_final.csv...")
    
    df = pd.read_csv('SOTU_data_final.csv', 
                     quotechar='"',
                     encoding='utf-8',
                     on_bad_lines='skip')
    
    print(f"Total speeches: {len(df)}\n")
    
    # Check Year column data type
    print("="*60)
    print("Year column information:")
    print("="*60)
    print(f"Data type: {df['Year'].dtype}")
    print(f"Unique years: {len(df['Year'].unique())}")
    print(f"Min year: {df['Year'].min()}")
    print(f"Max year: {df['Year'].max()}")
    print()
    
    # Show all rows between index positions that might contain 2009-2013
    print("="*60)
    print("Looking for rows with years containing '2010' or '2011' in any form:")
    print("="*60)
    
    # Check if Year is being read as string
    df['Year_str'] = df['Year'].astype(str)
    
    has_2010 = df[df['Year_str'].str.contains('2010', na=False)]
    has_2011 = df[df['Year_str'].str.contains('2011', na=False)]
    
    print(f"\nRows containing '2010': {len(has_2010)}")
    for idx, row in has_2010.iterrows():
        print(f"  Row {idx}: Year={repr(row['Year'])}, President={row['President']}, Title={row['Title'][:60]}")
    
    print(f"\nRows containing '2011': {len(has_2011)}")
    for idx, row in has_2011.iterrows():
        print(f"  Row {idx}: Year={repr(row['Year'])}, President={row['President']}, Title={row['Title'][:60]}")
    
    # Show all unique year values sorted
    print("\n" + "="*60)
    print("All unique Year values (first 50):")
    print("="*60)
    unique_years = sorted(df['Year'].unique())
    for i, year in enumerate(unique_years[:50]):
        print(f"{year} (type: {type(year).__name__}, repr: {repr(year)})")
    
    # Look at rows around Obama's presidency
    print("\n" + "="*60)
    print("All Obama speeches:")
    print("="*60)
    obama = df[df['President'].str.contains('Obama', na=False)]
    for idx, row in obama.iterrows():
        print(f"Year: {repr(row['Year'])}, Title: {row['Title'][:60]}")

if __name__ == "__main__":
    main()
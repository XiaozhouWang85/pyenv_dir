WITH remapped AS (

  SELECT
  transactions_id,
  price,
  date_of_transfer,
  postcode,
  property_type,
  old_new,
  duration,
  paon,
  saon,
  street,
  locality,
  town_city,
  district,
  county,
  category,
  CASE record_status
    WHEN "A" THEN 1
    WHEN "C" THEN 2
    WHEN "D" THEN 3
    ELSE 99
  END record_status
  FROM `fashion-scraping.ppd.raw_events`
  
),

ranked AS (

  SELECT 
  *,
  ROW_NUMBER() OVER(trans_part) AS rank_rev
  FROM remapped
  WINDOW trans_part AS (
    PARTITION BY transactions_id ORDER BY record_status DESC
  )

)

SELECT 
* EXCEPT(record_status,rank_rev)
FROM ranked
WHERE rank_rev = 1
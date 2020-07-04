

/************** CREATE TABLE ******************/


--CREATE TABLE Agg_1m (
--	obs_id INT PRIMARY KEY IDENTITY (1, 1),
--    ticker varchar(20) NOT NULL,
--	time_date date NOT NULL,
--	time_group int NOT NULL,
--	open_price decimal (38,5) NOT NULL,
--	high_price decimal (38,5) NOT NULL,
--	low_price decimal (38,5) NOT NULL,
--	close_price decimal (38,5) NOT NULL,
--	spread decimal (38,5)
--);

CREATE TABLE Agg_1m (
	--obs_id INT PRIMARY KEY IDENTITY (1, 1),
    ticker varchar(20) NOT NULL,
	time_date date NOT NULL,
	time_group int NOT NULL,
	open_price decimal (38,5) NOT NULL,
	high_price decimal (38,5) NOT NULL,
	low_price decimal (38,5) NOT NULL,
	close_price decimal (38,5) NOT NULL,
	spread decimal (38,5)
	PRIMARY KEY (ticker, time_date, time_group)
);

--DROP TABLE Agg_1m

--ALTER TABLE Agg_1m
--ADD column_name data_type column_constraint;



/************** SELECT FROM, INSERT INTO TABLE ******************/


SELECT TOP (1000) --[obs_id]
	  [ticker]	
      ,[time_date]
      ,[time_group]
      ,[open_price]
      ,[high_price]
      ,[low_price]
      ,[close_price]
      ,[spread]
  FROM [dbo].[Agg_1m]



INSERT INTO [dbo].[Agg_1m] (
		--[obs_id]
	  [ticker]
      ,[time_date]
      ,[time_group]
      ,[open_price]
      ,[high_price]
      ,[low_price]
      ,[close_price]
      --,[spread]
)
VALUES
    (
	'GOOG',
	'20200401',
	0,
	1122.26,
	1127.22,
	1119.51,
	1126.12
    );


DELETE TOP (10000) FROM [dbo].[Agg_1m]




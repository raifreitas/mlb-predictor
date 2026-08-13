namespace PruebaPronosticos.ETL;

public static class EstadioCatalog
{
    public static readonly Dictionary<string, (double Lat, double Lon, int ParkFactor)> Estadios = new()
    {
        { "Yankee Stadium", (40.8296, -73.9262, 101) },
        { "Fenway Park", (42.3467, -71.0972, 104) },
        { "Oriole Park at Camden Yards", (39.2840, -76.6215, 99) },
        { "Tropicana Field", (27.7683, -82.6534, 98) },
        { "Rogers Centre", (43.6414, -79.3894, 99) },
        { "Guaranteed Rate Field", (41.8299, -87.6338, 101) },
        { "Rate Field", (41.8299, -87.6338, 101) },
        { "Progressive Field", (41.4962, -81.6852, 100) },
        { "Comerica Park", (42.3390, -83.0485, 97) },
        { "Kauffman Stadium", (39.0517, -94.4803, 102) },
        { "Target Field", (44.9817, -93.2778, 98) },
        { "Minute Maid Park", (29.7573, -95.3555, 99) },
        { "Angel Stadium", (33.8003, -117.8827, 98) },
        { "Oakland Coliseum", (37.7516, -122.2005, 95) },
        { "T-Mobile Park", (47.5914, -122.3325, 93) },
        { "Globe Life Field", (32.7373, -97.0844, 99) },
        { "Truist Park", (33.8907, -84.4677, 101) },
        { "loanDepot park", (25.7783, -80.2196, 94) },
        { "Citi Field", (40.7571, -73.8458, 97) },
        { "Citizens Bank Park", (39.9061, -75.1665, 102) },
        { "Nationals Park", (38.8730, -77.0074, 99) },
        { "Wrigley Field", (41.9484, -87.6553, 101) },
        { "Great American Ball Park", (39.0979, -84.5082, 105) },
        { "American Family Field", (43.0280, -87.9712, 100) },
        { "PNC Park", (40.4469, -80.0057, 98) },
        { "Busch Stadium", (38.6226, -90.1928, 97) },
        { "Chase Field", (33.4455, -112.0667, 100) },
        { "Coors Field", (39.7559, -104.9942, 115) },
        { "Dodger Stadium", (34.0739, -118.2400, 99) },
        { "UNIQLO Field at Dodger Stadium", (34.0739, -118.2400, 99) },
        { "Petco Park", (32.7076, -117.1570, 96) },
        { "Oracle Park", (37.7786, -122.3893, 97) },
        { "Sutter Health Park", (38.5804, -121.5135, 100) }
    };
}

using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class teleport_tube : MonoBehaviour
{
    public GameObject player;
    public GameObject teleport_canvas;

    // Start is called before the first frame update
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        GameObject teleport_disk = transform.Find("teleport_disk").gameObject;
        Debug.Log(teleport_disk);

        // If player enters disk, turn on controller desk
        if (enter_disk(player, teleport_disk)){
            teleport_canvas.SetActive(true);
        }    
    }

    // Method for checking if player enters disk
    bool enter_disk(GameObject player, GameObject disk){
        Vector3 position1 =  player.transform.position;
        Vector3 position2 =  disk.transform.position;
        Vector3 disk_size = disk.transform.GetComponent<Renderer>().bounds.size;

        float x_distance = Mathf.Abs(position1.x - position2.x);
        float z_distance = Mathf.Abs(position1.z - position2.z);

        Debug.Log("x_distance: " + x_distance.ToString() + " / " + "z_distance: " + z_distance.ToString() + " / " + "disk_size: " + disk_size.ToString());

        if ( (x_distance < (disk_size.x/2)) && (z_distance < (disk_size.z/2)) ){
            return true;
        }
        else{
            return false;
        }
    }    
}
